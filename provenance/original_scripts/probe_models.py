import argparse
import gc
import time
import numpy as np
import torch
from common import *
from input_grid import variants
from model_adapters import build,output,sync


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=MODELS);args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261106);np.random.seed(20261106)
    rows=read(CORE/'runs/shortform_pilot_20261106/technical_manifest.csv')
    cache=SPEECH/'data/derived/radar_shortform_20261106'
    inputs=np.stack([v for row in rows for _,v in variants(np.load(cache/(row['utterance_id']+'.npy'),allow_pickle=False))])
    assert inputs.shape==(344,64600)
    results={};outputs={}
    for device in ('cpu','mps'):
        model,path=build(args.model,device);scores=[]
        # Warmup excluded from timing; every input still audited for finiteness.
        with torch.inference_mode():output(model,args.model,torch.from_numpy(inputs[:8]).to(device))
        sync(device);start=time.perf_counter()
        with torch.inference_mode():
            for i in range(0,len(inputs),8):
                s,pair=output(model,args.model,torch.from_numpy(inputs[i:i+8]).to(device));scores.extend(s.cpu().tolist())
        sync(device);elapsed=time.perf_counter()-start
        outputs[device]=np.array(scores);results[device]=dict(seconds=elapsed,seconds_per_input=elapsed/len(inputs),batch_size=8,finite=True,checkpoint_sha256=sha(path))
        del model;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    difference=np.abs(outputs['cpu']-outputs['mps'])
    decision='mps'
    if args.model in ('rawnet2','antideepfake') and difference.max()>1e-3:decision='cpu'
    result=dict(model=args.model,n_excluded_technical_files=8,n_inputs=344,torch=torch.__version__,
        cpu_vs_mps_max_abs_score_difference=float(difference.max()),median_abs_score_difference=float(np.median(difference)),
        n_differences_over_1e_3=int(np.sum(difference>1e-3)),runtimes=results,selected_device=decision,
        runtime_rule='Legacy MPS retained with known numerical sensitivity; new model uses MPS only if technical max difference<=1e-3, otherwise CPU. No accuracy-based selection.',
        no_scientific_metrics_computed=True)
    dump(ROOT/'sources'/f'probe_{args.model}.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
