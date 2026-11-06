import collections
import math
import numpy as np
from common import *
from input_grid import variants


def main():
    frozen=load(ROOT/'frozen_run.json');proof={};total=0
    for group in ('calibration','vi','itw'):
        manifest=ROOT/'calibration_manifest.csv' if group=='calibration' else ROOT/group/'manifest.csv';rows=read(manifest)
        cache=SPEECH/'data/derived/radar_shortform_20261106' if group=='calibration' else CACHE/group
        metadata={}
        for r in rows:
            x=np.load(cache/(r['utterance_id']+'.npy'),allow_pickle=False)
            metadata[r['utterance_id']]={v['condition']:v for v,_ in variants(x) if group!='calibration' or v['condition']=='official'}
        lookup={r['utterance_id']:r for r in rows};expected=1 if group=='calibration' else 43
        for model in MODELS:
            path=ROOT/'results'/f'{group}_{model}.csv';observed=set();count=0
            for r in read(path):
                uid=r['utterance_id'];condition=r['condition'];key=(uid,condition)
                assert uid in lookup and key not in observed;observed.add(key)
                assert r['label']==lookup[uid]['label'] and r['language']==lookup[uid]['language']
                v=metadata[uid][condition]
                for field in v:
                    if isinstance(v[field],str):assert r[field]==v[field]
                    else:assert float(r[field])==v[field],(model,uid,condition,field)
                score,a,b=[float(r[k]) for k in ('score','output0','output1')];assert all(math.isfinite(x) for x in (score,a,b))
                derived=-b if model in ('aasist','rawnet2') else (float(np.float32(a)-np.float32(b)) if model=='antideepfake' else a)
                assert abs(score-derived)<=1e-5,(model,uid,condition,'polarity')
                count+=1
            assert count==len(rows)*expected
            run=load(path.with_suffix('.run.json'))
            assert run['checkpoint_sha256']==frozen['models'][model]['checkpoint_sha256'] and run['manifest_sha256']==sha(manifest)
            assert run['all_modules_eval'] and run['device']==frozen['models'][model]['device']
            proof[group+'_'+model]=dict(files=len(rows),conditions=expected,rows=count,all_ID_label_offsets_input_placement_score_checks_passed=True);total+=count
    assert total==552000,total
    dump(ROOT/'results/score_audit.json',dict(total_rows=total,checks=proof,all_passed=True));print('Independent scientific-row audit passed:',total,flush=True)


if __name__=='__main__':main()
