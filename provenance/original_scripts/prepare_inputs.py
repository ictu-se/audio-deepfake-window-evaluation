import argparse
import hashlib
import importlib.util
import re
import tarfile
import numpy as np
import librosa
import soundfile as sf
from common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('group',choices=['vi','itw']);args=parser.parse_args();group=args.group
    plan=load(ROOT/'selection_plan.json');assert sha(ROOT/group/'candidate_manifest.csv')==plan['candidate_sha256'][group]
    rows=read(ROOT/group/'candidate_manifest.csv');raw=RAW/group;raw.mkdir(parents=True,exist_ok=True)
    if group=='itw':
        path=CORE/'external_itw_20261106/download_selected.py'
        spec=importlib.util.spec_from_file_location('pinned_itw_downloader',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        mod.HERE=ROOT/'itw';mod.RAW=raw;mod.main()
    else:
        archive_dir=SPEECH/'data/raw/radar2026'
        prior=load(archive_dir/'download_status.json');assert prior['complete'] and prior['md5']=='008b8c10bae684057d7f9d43bff2b55d'
        wanted={r['utterance_id'] for r in rows};found=set()
        with tarfile.open(archive_dir/'RADAR2026-eval.tar.gz','r|gz') as archive:
            for member in archive:
                uid=Path(member.name).stem
                if not member.isfile() or not member.name.endswith('.flac') or uid not in wanted:continue
                body=archive.extractfile(member).read();assert len(body)==member.size
                path=raw/(uid+'.flac')
                if path.exists():assert path.read_bytes()==body
                else:path.write_bytes(body)
                assert uid not in found;found.add(uid)
                if len(found)%200==0:print('VI source members verified',len(found),len(wanted),flush=True)
        assert found==wanted
    seen=set()
    for directory in (SPEECH/'data/derived/radar_shortform_20261106',SPEECH/'data/derived/itw_shortform_20261106',CACHE/'vi',CACHE/'itw'):
        if not directory.exists():continue
        for p in directory.glob('*.json'):
            if directory==CACHE/group:continue
            j=load(p)
            if 'pcm_sha256' in j:seen.add(j['pcm_sha256'])
    selected=[];excluded=[];counts={'bonafide':0,'spoof':0};target=plan['targets'][group];cache=CACHE/group;cache.mkdir(parents=True,exist_ok=True)
    for row in rows:
        uid,label=row['utterance_id'],row['label']
        if counts[label]>=target[label]:continue
        source=raw/(uid+'.flac' if group=='vi' else row['file']);info=sf.info(source)
        audio,sr=librosa.load(str(source),sr=16000,mono=True,dtype=np.float32,res_type='kaiser_best')
        valid=len(audio)>=64600 and np.isfinite(audio).all() and abs(info.duration-float(row['metadata_duration_s']))<=.01
        if group=='vi':
            match=re.fullmatch(r'([0-9.]+)-([0-9.]+)-(bonafide|spoof)',row['segments'])
            valid=valid and match is not None and float(match[1])==0 and match[3]==label and abs(float(match[2])-float(row['metadata_duration_s']))<=.001
        else:
            proof=load(source.with_suffix('.json'));assert proof['sha256']==sha(source)
            valid=valid and info.samplerate==16000 and info.channels==1 and info.subtype=='FLOAT' and source.stat().st_size==58+4*info.frames
        if not valid:excluded.append(dict(utterance_id=uid,reason='predefined finite/length/format/duration/segment check'));continue
        pcm=hashlib.sha256(audio.tobytes()).hexdigest()
        if pcm in seen:excluded.append(dict(utterance_id=uid,reason='exact PCM duplicate'));continue
        seen.add(pcm);path=cache/(uid+'.npy');np.save(path,audio,allow_pickle=False)
        dump(path.with_suffix('.json'),dict(source_sha256=sha(source),pcm_sha256=pcm,numpy_file_sha256=sha(path),frames=len(audio),source_samplerate=info.samplerate,source_frames=info.frames))
        selected.append({k:row[k] for k in ('utterance_id','language','label','metadata_duration_s')+ (('speaker_group',) if group=='itw' else ())})
        counts[label]+=1
        if len(selected)%100==0:print(group,'audited',len(selected),'counts',counts,flush=True)
    assert counts==target,(counts,target)
    selected.sort(key=lambda r:r['utterance_id']);manifest=ROOT/group/'manifest.csv';write_csv(manifest,selected)
    dump(ROOT/group/'input_audit.json',dict(n=len(selected),counts=counts,manifest_sha256=sha(manifest),exclusions=excluded,
        exact_PCM_duplicates=0,all_sources_and_caches_hashed=True,actual_format_duration_finite_checks_passed=True,
        speaker_groups=len({r['speaker_group'] for r in selected}) if group=='itw' else None,scored_before_manifest=False))
    print(group,'INPUTS COMPLETE',len(selected),flush=True)


if __name__=='__main__':main()
