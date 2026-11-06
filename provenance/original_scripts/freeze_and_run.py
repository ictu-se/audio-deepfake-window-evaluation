import datetime
import fcntl
import os
import shutil
import subprocess
import sys
import traceback
from common import *


def status(stage,**data):
    path=ROOT/'status.tmp';dump(path,dict(stage=stage,pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data));os.replace(path,ROOT/'status.json')
    print(stage,data,flush=True)


def main():
    lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    py=Path(__import__('sys').executable)
    seen=set()
    for directory in (SPEECH/'data/derived/radar_shortform_20261106',SPEECH/'data/derived/itw_shortform_20261106'):
        seen.update(load(p)['pcm_sha256'] for p in directory.glob('*.json') if 'pcm_sha256' in load(p))
    inputs={};cache_hashes={};audits={}
    for group in ('vi','itw'):
        audit=load(ROOT/group/'input_audit.json');assert sha(ROOT/group/'manifest.csv')==audit['manifest_sha256']
        vad=load(ROOT/group/'vad_audit.json');assert vad['files']==audit['n'] and vad['config_sha256']==sha(ROOT/'vad_config.json')
        rows=read(ROOT/group/'manifest.csv');cache_hashes[group]={}
        for row in rows:
            path=CACHE/group/(row['utterance_id']+'.npy');meta=load(path.with_suffix('.json'))
            assert meta['pcm_sha256'] not in seen,(group,row['utterance_id'],'PCM overlap');seen.add(meta['pcm_sha256'])
            assert sha(path)==meta['numpy_file_sha256'];cache_hashes[group][row['utterance_id']]=meta['numpy_file_sha256']
        inputs[group]=sha(ROOT/group/'manifest.csv');audits[group]=audit
    shutil.copy2(CORE/'runs/shortform_pilot_20261106/calibration_manifest.csv',ROOT/'calibration_manifest.csv')
    inputs['calibration']=sha(ROOT/'calibration_manifest.csv')
    probes={name:load(ROOT/'sources'/f'probe_{name}.json') for name in MODELS}
    models={name:dict(device=probes[name]['selected_device'],checkpoint_sha256=probes[name]['runtimes']['cpu']['checkpoint_sha256'],
        score={'aasist':'negative bona fide logit1','ssl':'fake logit0','rawnet2':'negative bona fide log-probability1','antideepfake':'fake logit0 minus real logit1 (monotonic fake softmax probability)'}[name],
        preprocessing='Per-input waveform layer normalization AFTER crop/padding, as in author model card' if name=='antideepfake' else 'Upstream model, no added global waveform normalization') for name in MODELS}
    code={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('*.py')}
    core_sources=load(CORE/'runs/shortform_pilot_20261106/frozen_main_config.json')['code_sha256']
    assert {p.name:sha(p) for p in CORE.glob('*.py')}==core_sources
    config=dict(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=models,code_sha256=code,
        original_core_code_sha256=core_sources,source_adapters_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'sources').glob('*.py')},
        manifest_sha256=inputs,cache_sha256=cache_hashes,input_audits=audits,vad_config_sha256=sha(ROOT/'vad_config.json'),
        vad_windows_sha256={g:sha(ROOT/g/'vad_windows.csv') for g in ('vi','itw')},selection_plan_sha256=sha(ROOT/'selection_plan.json'),
        grid='19 original conditions +12 additional ninth-offset repeated crops +12 identical-content input placements =43 conditions; common raw input64600 samples',
        primary='Middle-minus-start EER, mean1s/2s repeat, each of4 models on2 fresh cohorts;8 contrasts in one family,10000 paired draws,99.375% percentile intervals',
        primary_seeds={'vi':20261160,'itw':20261161},secondary_bootstrap_draws=1000,
        secondary='Full grid; matched-content center/right-minus-left placement; VAD common-file subset; five fixed policies; nominal95% operating-point intervals jointly resample calibration and test; no family-wide claims',
        policy_rules='start, middle, per-ID SHA256 random among9 offsets, mean over0/4/8, maximum predicted speech fraction with earliest tie; all fixed in design, no target-based policy selection',
        vad_amendment='Whole-unpadded-recording segmentation followed by interval intersection; parameters fixed in vad_config.json before VAD outputs',
        model_amendment='Res-TSSDNet failed native64600 pooling gate; replaced with official RawNet2 before any scientific scores. No architecture repair or EER-based exclusion.',
        expected_scientific_scores=552000,batch_size=8,float32=True,training=False,
        limits=['RADAR file-disjoint, hidden speaker/content/source dependence remains.',
            'ITW external to our RADAR development; upstream AntiDeepfake already reports ITW benchmark results. No claim upstream-unseen benchmark or training-population independence.',
            'Pipelines differ in architecture, training mixture and native preprocessing; not a causal architecture comparison.',
            'VAD predictions are not speech annotations; VAD-restricted cohorts change the population.',
            'Legacy SSL MPS has known score differences, including current technical probe; CIs quantify sampling, not execution-stack uncertainty.',
            'Only long-source crops, offline full-recording policies; no native short-utterance or streaming-latency claim.',
            'Runtime benchmarks use8 excluded technical files with warm OS cache, not a deployment latency study.'])
    path=ROOT/'frozen_run.json'
    if path.exists():
        old=load(path);config['locked_utc']=old['locked_utc'];assert config==old
    else:
        assert not any(list((ROOT/'results').glob('*_'+model+'.csv')) for model in MODELS)
        dump(path,config)
    status('locked',files={k:v['n'] for k,v in audits.items()},models=list(MODELS))
    for name in MODELS:
        assert all(sha(ROOT/p)==h for p,h in code.items())
        status('inference',model=name)
        with (ROOT/'logs'/f'inference_{name}.log').open('a') as log:
            subprocess.run([str(py),str(ROOT/'scripts/run_inference.py'),'--model',name],stdout=log,stderr=subprocess.STDOUT,check=True)
    for stage,script in (('independent_score_audit','audit_scores.py'),('policy_runtime_benchmark','benchmark_policies.py'),('bootstrap_analysis','analyze_expansion.py')):
        status(stage)
        with (ROOT/'logs'/(stage+'.log')).open('a') as log:
            subprocess.run([str(py),str(ROOT/'scripts'/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
    assert all(sha(ROOT/p)==h for p,h in code.items())
    assert {p.name:sha(p) for p in CORE.glob('*.py')}==core_sources
    status('complete',scientific_scores=552000,summary='results/expansion_summary.json',old_scientific_results_unchanged=True)
    files=[p for p in (ROOT/'results').iterdir() if p.is_file() and p.suffix in ('.csv','.json','.npy')]
    dump(ROOT/'results/artifact_sha256.json',{p.name:sha(p) for p in files if p.name!='artifact_sha256.json'})


if __name__=='__main__':
    try:main()
    except Exception:
        error=traceback.format_exc();(ROOT/'error.txt').write_text(error);status('error',message=error[-2200:]);raise
