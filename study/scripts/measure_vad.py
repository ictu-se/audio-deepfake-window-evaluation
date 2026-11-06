import argparse
import importlib.util
import time
import numpy as np
import torch
from common import *

VAD_SETTINGS=dict(threshold=.5,neg_threshold=.35,sampling_rate=16000,min_speech_duration_ms=250,min_silence_duration_ms=100,speech_pad_ms=0,return_seconds=False)


def build_vad():
    path=ROOT/'sources/silero_utils_vad.py';spec=importlib.util.spec_from_file_location('expanded_silero_utils',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=torch.jit.load(str(ROOT/'models/silero_vad.jit'),map_location='cpu').eval()
    return model,mod.get_speech_timestamps


def coverage(segments,first,last):
    return sum(max(0,min(last,s['end'])-max(first,s['start'])) for s in segments)/(last-first)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('group',choices=['vi','itw']);args=parser.parse_args();g=args.group
    assert load(ROOT/g/'input_audit.json')['manifest_sha256']==sha(ROOT/g/'manifest.csv')
    config=load(ROOT/'vad_config.json');assert config['settings']==VAD_SETTINGS
    torch.set_num_threads(1);model,function=build_vad();rows=read(ROOT/g/'manifest.csv');out=[];proof=[]
    started=time.perf_counter()
    for i,row in enumerate(rows):
        uid=row['utterance_id'];path=CACHE/g/(uid+'.npy');x=np.load(path,allow_pickle=False)
        assert sha(path)==load(path.with_suffix('.json'))['numpy_file_sha256']
        t=time.perf_counter()
        with torch.inference_mode():segments=function(torch.from_numpy(x),model,**VAD_SETTINGS)
        elapsed=time.perf_counter()-t
        proof.append(dict(utterance_id=uid,source_frames=len(x),vad_seconds=elapsed,speech_segments=segments))
        for duration in (1,2):
            size=duration*16000
            for k in range(9):
                offset=((len(x)-size)*k)//8
                fraction=coverage(segments,offset,offset+size)
                assert 0<=fraction<=1
                out.append(dict(utterance_id=uid,label=row['label'],duration_s=duration,source_eighths=k,source_offset=offset,speech_fraction=fraction))
        if (i+1)%100==0:print('VAD',g,i+1,'/',len(rows),'seconds',round(time.perf_counter()-started,1),flush=True)
    write_csv(ROOT/g/'vad_windows.csv',out);dump(ROOT/g/'vad_segments.json',proof)
    dump(ROOT/g/'vad_audit.json',dict(files=len(rows),windows=len(out),waveforms_unpadded=True,full_recording_segmentation=True,
        crop_fraction='Intersection of predicted full-recording speech intervals with each unpadded crop; not VAD ground truth',
        no_detector_scores_used=True,settings=VAD_SETTINGS,model_sha256=sha(ROOT/'models/silero_vad.jit'),config_sha256=sha(ROOT/'vad_config.json')))


if __name__=='__main__':main()
