"""Grouped-score ROC metrics and paired, class-stratified file bootstrap."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve

HERE=Path(__file__).resolve().parent



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

