import gc
import time
import numpy as np
import torch
import librosa
import resource
from common import *
from input_grid import repeat_name
from measure_vad import build_vad,coverage,VAD_SETTINGS
from model_adapters import build,output,sync


def main():
    frozen=load(ROOT/'frozen_run.json');torch.set_num_threads(1)
    technical=read(CORE/'runs/shortform_pilot_20261106/technical_manifest.csv')
    vad,segment=build_vad();records=[]
    for model_name in MODELS:
        device=frozen['models'][model_name]['device'];model,path=build(model_name,device)
        # Warm models and the first file only. Cache state and shared-host load are uncontrolled.
        source=RAW/'technical'/(technical[0]['utterance_id']+'.flac')
        warm,_=librosa.load(str(source),sr=16000,mono=True,dtype=np.float32,res_type='kaiser_best')
        with torch.inference_mode():
            for batchsize in (1,3):output(model,model_name,torch.from_numpy(np.tile(warm[:64600][None,:],(batchsize,1))).to(device))
            segment(torch.from_numpy(warm),vad,**VAD_SETTINGS)
        sync(device)
        for repetition in range(3):
            for row in technical:
                source=source.with_name(row['utterance_id']+'.flac')
                for duration in (1,2):
                    for policy in ('start','middle','random','mean_three','vad_max'):
                        sync(device);started=time.perf_counter()
                        x,_=librosa.load(str(source),sr=16000,mono=True,dtype=np.float32,res_type='kaiser_best');size=duration*16000
                        vad_seconds=0.0
                        if policy=='vad_max':
                            t=time.perf_counter()
                            with torch.inference_mode():segments=segment(torch.from_numpy(x),vad,**VAD_SETTINGS)
                            vad_seconds=time.perf_counter()-t
                            keys=[max(range(9),key=lambda k:coverage(segments,((len(x)-size)*k)//8,((len(x)-size)*k)//8+size))]
                        elif policy=='mean_three':keys=[0,4,8]
                        elif policy=='random':keys=[int(__import__('hashlib').sha256(('expansion-random-20261106:'+row['utterance_id']).encode()).hexdigest(),16)%9]
                        else:keys=[0 if policy=='start' else 4]
                        inputs=[]
                        for k in keys:
                            off=((len(x)-size)*k)//8;crop=x[off:off+size];inputs.append(np.tile(crop,int(np.ceil(64600/size)))[:64600])
                        batch=torch.from_numpy(np.stack(inputs)).to(device);sync(device);t=time.perf_counter()
                        with torch.inference_mode():output(model,model_name,batch)
                        sync(device);model_seconds=time.perf_counter()-t;elapsed=time.perf_counter()-started
                        records.append(dict(model=model_name,device=device,utterance_id=row['utterance_id'],duration_s=duration,policy=policy,repetition=repetition,
                            total_seconds=elapsed,model_seconds=model_seconds,vad_seconds=vad_seconds,detector_inputs=len(keys),detector_batch_calls=1,
                            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            mps_driver_bytes=torch.mps.driver_allocated_memory() if device=='mps' else 0))
        del model;gc.collect();torch.mps.empty_cache();print('Policy benchmark complete',model_name,flush=True)
    write_csv(ROOT/'results/policy_runtime.csv',records)
    dump(ROOT/'results/policy_runtime_scope.json',dict(technical_files=8,repetitions=3,labels_or_accuracy_used=False,torch_threads=1,
        benchmark='Warm-cache raw FLAC decode/resample, crop selection, optional full-recording VAD, classifier. Mean-three batches 3 inputs in one model call; other policies batch1.',
        model_only_time='Excludes decode/VAD/device transfer; synchronized around classifier computation',
        memory='Per-process historical max RSS on macOS plus current MPS driver allocation; not isolated per-policy peak memory',
        streaming_claim=False,clinical_or_deployment_latency_claim=False))


if __name__=='__main__':main()
