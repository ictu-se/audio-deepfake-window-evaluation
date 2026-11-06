"""Grouped-score ROC metrics and paired, class-stratified file bootstrap."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/shortform_pilot_20261106'


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def metric(y,s,w=None):
    fpr,tpr,threshold=roc_curve(y,s,sample_weight=w,drop_intermediate=False)
    k=int(np.argmin(np.abs(fpr-(1-tpr))))
    return {'eer':float((fpr[k]+1-tpr[k])/2),'auc':float(roc_auc_score(y,s,sample_weight=w)),
            'eer_threshold':float(threshold[k])}


class WeightedEER:
    def __init__(self,y,s):
        self.order=np.argsort(-np.asarray(s),kind='stable')
        sorted_s=np.asarray(s)[self.order]
        self.ends=np.r_[np.flatnonzero(np.diff(sorted_s)),len(s)-1]
        self.y=np.asarray(y)[self.order]
    def __call__(self,weights):
        w=np.asarray(weights)[self.order]
        pos=w*self.y;neg=w*(1-self.y)
        assert pos.sum()>0 and neg.sum()>0
        tpr=np.r_[0,np.cumsum(pos)[self.ends]/pos.sum()]
        fpr=np.r_[0,np.cumsum(neg)[self.ends]/neg.sum()]
        k=int(np.argmin(np.abs(fpr-(1-tpr))))
        return float((fpr[k]+1-tpr[k])/2)


def matrix(path,manifest,expected=19):
    people=read(manifest);ids=[r['utterance_id'] for r in people];labels={r['utterance_id']:r['label'] for r in people}
    assert len(set(ids))==len(ids)
    table={}
    for row in read(path):
        uid=row['utterance_id'];assert uid in labels and row['label']==labels[uid]
        condition=row['condition'];table.setdefault(condition,{})
        assert uid not in table[condition];table[condition][uid]=float(row['score'])
    assert len(table)==expected,(path,len(table))
    for cells in table.values():assert set(cells)==set(ids)
    scores={c:np.array([cells[uid] for uid in ids]) for c,cells in table.items()}
    assert all(np.isfinite(v).all() for v in scores.values())
    y=np.array([labels[uid]=='spoof' for uid in ids],dtype=int)
    return ids,y,scores


def add_aggregates(scores):
    result=dict(scores)
    for d in [1,2,4]:
        for pad in ['repeat','zero']:
            a=np.stack([scores[f'{d}s_{p}_{pad}'] for p in ['start','middle','end']])
            for name,func in [('mean',np.mean),('median',np.median),('max',np.max)]:
                result[f'{d}s_aggregate_{name}_{pad}']=func(a,axis=0)
    return result


def primary(y,scores,draws=2000):
    cells=[(f'{d}s_{p}_zero',f'{d}s_{p}_repeat') for d in [1,2] for p in ['start','middle','end']]
    calculators={c:WeightedEER(y,scores[c]) for pair in cells for c in pair}
    def contrast(w):return float(np.mean([calculators[a](w)-calculators[b](w) for a,b in cells]))
    estimate=contrast(np.ones(len(y)))
    rng=np.random.default_rng(20261106);classes=[np.flatnonzero(y==c) for c in [0,1]]
    sampled=[]
    for _ in range(draws):
        w=np.zeros(len(y),dtype=int)
        for group in classes:w[group]=rng.multinomial(len(group),np.full(len(group),1/len(group)))
        sampled.append(contrast(w))
    lo,hi=np.quantile(sampled,[.0125,.9875])
    return {'contrast':'mean short-duration EER(zero)-EER(repeat), averaged over 1/2s and start/middle/end',
            'estimate_percentage_points':estimate*100,'interval_97_5_percentage_points':[float(lo*100),float(hi*100)],
            'bootstrap_draws':draws,'bootstrap_unit':'released file, class-stratified; speaker/content grouping unavailable',
            'multiplicity':'97.5% interval per each of two prespecified model contrasts; secondary cell analyses descriptive'},np.asarray(sampled)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','main'],required=True);ap.add_argument('--draws',type=int,default=2000);args=ap.parse_args()
    results={};rows=[]
    for model in ['aasist','ssl']:
        prefix='pilot' if args.phase=='pilot' else 'main_vi'
        manifest=RUN/('pilot_manifest.csv' if args.phase=='pilot' else 'main_vi_manifest.csv')
        ids,y,scores=matrix(RUN/f'{prefix}_{model}.csv',manifest)
        threshold=None
        if args.phase=='main':
            _,cy,cs=matrix(RUN/f'calibration_{model}.csv',RUN/'calibration_manifest.csv',expected=1)
            threshold=metric(cy,cs['official'])['eer_threshold']
        for c,s in add_aggregates(scores).items():
            values=metric(y,s)
            row={'model':model,'dataset':prefix,'condition':c,'n':len(y),'eer_percent':values['eer']*100,'auc':values['auc']}
            if threshold is not None:
                row.update(threshold=threshold,bonafide_block_percent=float(np.mean(s[y==0]>=threshold)*100),spoof_pass_percent=float(np.mean(s[y==1]<threshold)*100))
            rows.append(row)
        if args.phase=='main':
            result,draws=primary(y,scores,args.draws);np.save(RUN/f'bootstrap_primary_{model}.npy',draws,allow_pickle=False)
            result['threshold_from_calibration']=threshold
            result['official']=metric(y,scores['official'])
            results[model]=result
            enids,ey,es=matrix(RUN/f'secondary_en_{model}.csv',RUN/'secondary_en_manifest.csv')
            for c,s in add_aggregates(es).items():
                values=metric(ey,s);rows.append({'model':model,'dataset':'secondary_en','condition':c,'n':len(ey),'eer_percent':values['eer']*100,'auc':values['auc'],
                    'threshold':threshold,'bonafide_block_percent':float(np.mean(s[ey==0]>=threshold)*100),'spoof_pass_percent':float(np.mean(s[ey==1]<threshold)*100)})
        else:results[model]={'n':len(y),'official':metric(y,scores['official']),'interpretation':'technical/exploratory pilot, excluded from main; not a confirmatory contribution'}
    fields=list(rows[0])
    with (RUN/f'{args.phase}_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (RUN/f'{args.phase}_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
