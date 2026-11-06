"""Frozen-weight waveform experiments; no fitting or checkpoint selection.

All offsets/padding are explicit. Technical synthetic probes are kept out of the
dataset, manifests and reported detection results.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
SPEECH=HERE.parents[1]
RUN=HERE/'runs/shortform_pilot_20261106'
RAW=SPEECH/'data/raw/radar2026/RADAR2026-eval/flac'
CACHE=SPEECH/'data/derived/radar_shortform_20261106'
RATE=16000
SAMPLES=64600


def load_manifest(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def variants(x):
    assert x.ndim==1 and len(x)>=SAMPLES
    yield dict(condition='official',duration_s=SAMPLES/RATE,position='start',padding='none',offset_samples=0,original_samples=SAMPLES),x[:SAMPLES]
    for duration in [1,2,4]:
        n=duration*RATE
        for position,offset in [('start',0),('middle',(len(x)-n)//2),('end',len(x)-n)]:
            part=x[offset:offset+n]
            assert len(part)==n and np.isfinite(part).all()
            for padding in ['repeat','zero']:
                out=np.tile(part,int(np.ceil(SAMPLES/n)))[:SAMPLES] if padding=='repeat' else np.pad(part,(0,SAMPLES-n))
                assert len(out)==SAMPLES and np.array_equal(out[:n],part)
                yield dict(condition=f'{duration}s_{position}_{padding}',duration_s=duration,position=position,padding=padding,offset_samples=offset,original_samples=n),np.asarray(out,dtype=np.float32)


def cache_audio(rows):
    import librosa
    import soundfile as sf
    CACHE.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        uid=row['utterance_id'];source=RAW/(uid+'.flac');target=CACHE/(uid+'.npy');meta=CACHE/(uid+'.json')
        if target.exists() and meta.exists():continue
        info=sf.info(source)
        # Matches the official inference loader, including resampling/mono policy.
        x,sr=librosa.load(str(source),sr=RATE,mono=True,dtype=np.float32,res_type='kaiser_best')
        assert sr==RATE and len(x)>=SAMPLES and np.isfinite(x).all(),uid
        np.save(target,x,allow_pickle=False)
        meta.write_text(json.dumps({'source_sha256':sha(source),'pcm_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                                   'source_samplerate':info.samplerate,'source_channels':info.channels,'source_frames':info.frames,
                                   'resampled_frames':len(x),'resampled_duration_s':len(x)/RATE,
                                   'metadata_duration_s':float(row['metadata_duration_s']),'librosa':librosa.__version__,
                                   'soundfile':sf.__version__,'resample':'kaiser_best','numpy_file_sha256':sha(target)},indent=2)+'\n')
        if i%25==0:print('cached',i+1,len(rows),uid,flush=True)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def build_model(name,device):
    if name=='aasist':
        base=HERE/'vendor/aasist';mod=module('aasist_arch',base/'models/AASIST.py')
        config=json.loads((base/'config/AASIST.conf').read_text())['model_config']
        model=mod.Model(config);checkpoint=base/'models/weights/AASIST.pth'
        state=torch.load(checkpoint,map_location='cpu');model.load_state_dict(state,strict=True)
    else:
        sys.path.insert(0,str(HERE/'vendor/fairseq_v0122'))
        mod=module('radar_arch',HERE/'vendor/radar/model.py')
        from types import SimpleNamespace
        base=HERE/'checkpoints/xlsr2_300m_runtime.pt'
        if not base.exists():base=HERE/'checkpoints/xlsr2_300m.pt'
        model=mod.Model(SimpleNamespace(ssl_cp_path=str(base)),'cpu')
        checkpoint=HERE/'checkpoints/Best_LA_model_for_DF.pth'
        state=torch.load(checkpoint,map_location='cpu')
        if 'state_dict' in state:state=state['state_dict']
        state={k.replace('module.','',1):v for k,v in state.items()}
        model.load_state_dict(state,strict=True)
    model.to(device).eval()
    assert not any(m.training for m in model.modules())
    return model,checkpoint

