import argparse
import csv
import fcntl
import time
import numpy as np
import torch
from common import *
from input_grid import variants
from model_adapters import build,output,sync


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=MODELS);args=parser.parse_args();name=args.model
    frozen=load(ROOT/'frozen_run.json')
    for path,h in frozen['code_sha256'].items():assert sha(ROOT/path)==h,path
    config=frozen['models'][name];device=config['device'];batchsize=8
    torch.set_num_threads(4);torch.manual_seed(20261106);np.random.seed(20261106)
    lock=(ROOT/'results'/f'{name}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model,checkpoint=build(name,device);assert sha(checkpoint)==config['checkpoint_sha256']
    for group in ('calibration','vi','itw'):
        manifest=ROOT/'calibration_manifest.csv' if group=='calibration' else ROOT/group/'manifest.csv'
        assert sha(manifest)==frozen['manifest_sha256'][group]
        rows=read(manifest);cache=SPEECH/'data/derived/radar_shortform_20261106' if group=='calibration' else CACHE/group
        path=ROOT/'results'/f'{group}_{name}.csv';done=set();had_header=path.exists() and path.stat().st_size>0
        if had_header:
            for row in read(path):
                key=(row['utterance_id'],row['condition']);assert key not in done;done.add(key)
        fields=['utterance_id','language','label','condition','duration_s','source_eighths','source_offset','input_offset','padding','payload_samples','score','output0','output1']
        started=time.perf_counter();count=0;pending=[];forward_seconds=0
        with path.open('a',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            if not had_header:writer.writeheader()
            def flush():
                nonlocal pending,count,forward_seconds
                if not pending:return
                batch=torch.from_numpy(np.stack([a for _,_,a in pending])).to(device)
                sync(device);t=time.perf_counter()
                with torch.inference_mode():score,pair=output(model,name,batch)
                sync(device);forward_seconds+=time.perf_counter()-t
                score=score.cpu().numpy();pair=pair.cpu().numpy()
                for (row,variant,_),s,p in zip(pending,score,pair):
                    writer.writerow(dict(**{k:row[k] for k in ('utterance_id','language','label')},**variant,score=float(s),output0=float(p[0]),output1=float(p[1])))
                    count+=1
                stream.flush();pending=[]
            for i,row in enumerate(rows):
                file=cache/(row['utterance_id']+'.npy')
                if group!='calibration':assert sha(file)==frozen['cache_sha256'][group][row['utterance_id']],row['utterance_id']
                x=np.load(file,allow_pickle=False)
                for info,wave in variants(x):
                    if group=='calibration' and info['condition']!='official':continue
                    if (row['utterance_id'],info['condition']) in done:continue
                    pending.append((row,info,wave))
                    if len(pending)==batchsize:flush()
                if (i+1)%50==0:print(name,group,i+1,'/',len(rows),'new_scores',count,'seconds',round(time.perf_counter()-started,1),flush=True)
            flush()
        expected=len(rows)*(1 if group=='calibration' else 43)
        assert len(done)+count==expected
        dump(path.with_suffix('.run.json'),dict(model=name,group=group,device=device,torch=torch.__version__,batch_size=batchsize,threads=4,
            checkpoint_sha256=sha(checkpoint),manifest_sha256=sha(manifest),expected_scores=expected,new_scores=count,previous_scores=len(done),
            forward_seconds=forward_seconds,wall_seconds=time.perf_counter()-started,score_semantics=config['score'],all_modules_eval=not any(m.training for m in model.modules())))
        print(name,group,'COMPLETE',expected,flush=True)


if __name__=='__main__':main()
