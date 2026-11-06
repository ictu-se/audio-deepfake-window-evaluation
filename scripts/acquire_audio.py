"""Acquire only the frozen recordings after checking their original access terms.
RADAR archive (8.75 GB) must already be downloaded from the cited release.
"""
import argparse,hashlib,importlib.util,json,tarfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'study/scripts'))
from common import read,sha,dump

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--radar-archive',type=Path,required=True);args=parser.parse_args()
 digest=hashlib.md5()
 with args.radar_archive.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
 assert digest.hexdigest()=='008b8c10bae684057d7f9d43bff2b55d','RADAR archive MD5 mismatch'
 sets={'vi':read(ROOT/'study/vi/manifest.csv'),'calibration':read(ROOT/'study/calibration_manifest.csv'),'technical':read(ROOT/'core/runs/shortform_pilot_20261106/technical_manifest.csv')}
 wanted={r['utterance_id']:g for g,rows in sets.items() for r in rows};found=set()
 with tarfile.open(args.radar_archive,'r|gz') as arc:
  for member in arc:
   uid=Path(member.name).stem
   if uid not in wanted or not member.isfile() or not member.name.endswith('.flac'):continue
   target=ROOT/'workspace/audio'/wanted[uid]/(uid+'.flac');target.parent.mkdir(parents=True,exist_ok=True)
   target.write_bytes(arc.extractfile(member).read());found.add(uid)
 assert found==set(wanted)
 path=ROOT/'core/external_itw_20261106/download_selected.py'
 spec=importlib.util.spec_from_file_location('download_itw',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.HERE=ROOT/'study/itw';m.RAW=ROOT/'workspace/audio/itw';m.main()
 frozen=json.loads((ROOT/'study/frozen_run.json').read_text());sets['itw']=read(ROOT/'study/itw/manifest.csv')
 for group,rows in sets.items():
  for r in rows:
   uid=r['utterance_id'];name=uid[4:]+'.wav' if group=='itw' else uid+'.flac'
   source=ROOT/'workspace/audio'/group/name
   audio,sr=librosa.load(str(source),sr=16000,mono=True,dtype=np.float32,res_type='kaiser_best')
   assert len(audio)>=64600 and np.isfinite(audio).all()
   target=ROOT/'workspace/cache'/group/(uid+'.npy');target.parent.mkdir(parents=True,exist_ok=True)
   np.save(target,audio,allow_pickle=False)
   if group in ('vi','itw'):assert sha(target)==frozen['cache_sha256'][group][uid],uid
   dump(target.with_suffix('.json'),{'numpy_file_sha256':sha(target),'source_sha256':sha(source)})
 print('Frozen recordings decoded; new-cohort cache hashes verified.')
if __name__=='__main__':main()
