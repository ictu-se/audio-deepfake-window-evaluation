import hashlib
import sys
import numpy as np
from common import *
from input_grid import repeat_name
sys.path.insert(0,str(CORE))
from analyze_shortform import matrix,metric,WeightedEER
from sklearn.metrics import roc_curve


def paired_weights(y,groups,seed,draws=1000):
    rng=np.random.default_rng(seed);out=[]
    if groups is not None:names,inverse=np.unique(groups,return_inverse=True)
    else:classes=[np.flatnonzero(y==v) for v in (0,1)]
    while len(out)<draws:
        if groups is None:
            w=np.zeros(len(y),dtype=np.int16)
            for ids in classes:w[ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
        else:w=rng.multinomial(len(names),np.full(len(names),1/len(names)))[inverse].astype(np.int16)
        if w[y==0].sum() and w[y==1].sum():out.append(w)
    return np.stack(out)


def threshold_draws(y,score):
    weights=paired_weights(y,None,20261162)
    thresholds=[]
    for w in weights:
        fpr,tpr,t=roc_curve(y,score,sample_weight=w,drop_intermediate=False)
        thresholds.append(t[int(np.argmin(abs(fpr-(1-tpr))))])
    return np.array(thresholds)


def joint_operating_intervals(y,score,weights,thresholds):
    blocked=score[None,:]>=thresholds[:,None]
    result={}
    for cls,name,error in ((0,'bonafide_block',blocked),(1,'spoof_pass',~blocked)):
        w=weights[:,y==cls];values=100*np.sum(w*error[:,y==cls],axis=1)/w.sum(axis=1)
        lo,hi=np.quantile(values,[.025,.975]);result[name+'_lo95']=float(lo);result[name+'_hi95']=float(hi)
    return result


def placement_uncertainty(y,s,weights):
    keys=[f'{d}s_{source}_zero'+suffix for d in (1,2) for source in ('start','middle','end') for suffix in ('','_center','_right')]
    calculators={k:WeightedEER(y,s[k]) for k in keys};out={}
    for placement in ('center','right'):
        def contrast(w):return 100*np.mean([calculators[f'{d}s_{source}_zero_{placement}'](w)-calculators[f'{d}s_{source}_zero'](w) for d in (1,2) for source in ('start','middle','end')])
        draws=np.array([contrast(w) for w in weights])
        out[placement+'_minus_left']=dict(estimate_pp=float(contrast(np.ones(len(y)))),nominal_interval_95_pp=np.quantile(draws,[.025,.975]).tolist(),
            bootstrap_draws=len(weights),interpretation='Predefined secondary paired contrast; identical payload and zero count; not in the eight-test primary family')
    return out


def primary(y,scores,groups,seed):
    keys=[f'{d}s_{p}_repeat' for d in (1,2) for p in ('start','middle')]
    calculators={c:WeightedEER(y,scores[c]) for c in keys}
    def contrast(w):return .5*sum(calculators[f'{d}s_middle_repeat'](w)-calculators[f'{d}s_start_repeat'](w) for d in (1,2))
    point=contrast(np.ones(len(y)));rng=np.random.default_rng(seed);draws=[]
    if groups is not None:
        names,inverse=np.unique(groups,return_inverse=True)
    else:classes=[np.flatnonzero(y==v) for v in (0,1)]
    rejected=0
    while len(draws)<10000:
        if groups is None:
            w=np.zeros(len(y),dtype=int)
            for indices in classes:w[indices]=rng.multinomial(len(indices),np.full(len(indices),1/len(indices)))
        else:w=rng.multinomial(len(names),np.full(len(names),1/len(names)))[inverse]
        if not w[y==0].sum() or not w[y==1].sum():rejected+=1;assert rejected<100;continue
        draws.append(contrast(w))
    a=np.array(draws);interval=100*np.quantile(a,[.003125,.996875])
    return dict(estimate_pp=float(100*point),interval_99_375_pp=interval.tolist(),bootstrap_draws=10000,seed=seed,
        bootstrap_unit='provided speaker group' if groups is not None else 'file, class-stratified',speaker_groups=len(names) if groups is not None else None,
        rejected_class_empty_draws=rejected,multiplicity='Eight model-by-cohort position contrasts, one family; nominal 99.375% percentile interval each'),a


def main():
    frozen=load(ROOT/'frozen_run.json');summary={'primary':{},'secondary':{},'placement_secondary':{},'limits':frozen['limits']};cells=[];policies=[];placements=[];profiles=[]
    for group in ('vi','itw'):
        rows=read(ROOT/group/'manifest.csv');byid={r['utterance_id']:r for r in rows}
        vad=read(ROOT/group/'vad_windows.csv');fractions={(r['utterance_id'],int(r['duration_s']),int(r['source_eighths'])):float(r['speech_fraction']) for r in vad}
        for model in MODELS:
            file=ROOT/'results'/f'{group}_{model}.csv';ids,y,s=matrix(file,ROOT/group/'manifest.csv',expected=43)
            groups=np.array([byid[uid]['speaker_group'] for uid in ids]) if group=='itw' else None
            secondary_weights=paired_weights(y,groups,20261163+(group=='itw'))
            result,draws=primary(y,s,groups,20261160+(group=='itw'))
            summary['primary'][group+'_'+model]=result;np.save(ROOT/'results'/f'bootstrap_{group}_{model}.npy',draws,allow_pickle=False)
            _,cy,cs=matrix(ROOT/'results'/f'calibration_{model}.csv',ROOT/'calibration_manifest.csv',expected=1)
            threshold=metric(cy,cs['official'])['eer_threshold']
            cal_thresholds=threshold_draws(cy,cs['official'])
            summary['placement_secondary'][group+'_'+model]=placement_uncertainty(y,s,secondary_weights)
            for condition,score in s.items():
                m=metric(y,score);cells.append(dict(cohort=group,model=model,condition=condition,eer_percent=100*m['eer'],auc=m['auc'],
                    calibration_threshold=threshold,bonafide_block_percent=100*float(np.mean(score[y==0]>=threshold)),spoof_pass_percent=100*float(np.mean(score[y==1]<threshold))))
            mask=np.array([all(fractions[uid,d,k]>=.8 for d in (1,2) for k in (0,4)) and
                all(abs(fractions[uid,d,0]-fractions[uid,d,4])<=.1 for d in (1,2)) for uid in ids])
            nr={str(c):int(np.sum(mask & (y==c))) for c in (0,1)}
            sensitivity=None
            if min(nr.values())>=30:sensitivity=50*sum(metric(y[mask],s[f'{d}s_middle_repeat'][mask])['eer']-metric(y[mask],s[f'{d}s_start_repeat'][mask])['eer'] for d in (1,2))
            summary['secondary'][group+'_'+model]=dict(vad_common_mask_retained=int(mask.sum()),retained_by_class=nr,vad_subset_position_contrast_pp=sensitivity,
                interpretation='Restricted population; predicted speech activity is not ground truth; secondary analysis without family-wide significance claim')
            for duration in (1,2):
                z=np.stack([s[repeat_name(duration,k)] for k in range(9)])
                best=np.array([max(range(9),key=lambda k:fractions[uid,duration,k]) for uid in ids])
                random=np.array([int(hashlib.sha256(('expansion-random-20261106:'+uid).encode()).hexdigest(),16)%9 for uid in ids])
                policy_scores={'start':z[0],'middle':z[4],'random':z[random,np.arange(len(ids))],
                    'mean_three':z[[0,4,8]].mean(axis=0),'vad_max':z[best,np.arange(len(ids))]}
                for name,score in policy_scores.items():
                    m=metric(y,score);policies.append(dict(cohort=group,model=model,duration_s=duration,policy=name,eer_percent=100*m['eer'],auc=m['auc'],
                        detector_forwards=3 if name=='mean_three' else 1,full_recording_vad_passes=1 if name=='vad_max' else 0,
                        bonafide_block_percent=100*float(np.mean(score[y==0]>=threshold)),spoof_pass_percent=100*float(np.mean(score[y==1]<threshold)),
                        **joint_operating_intervals(y,score,secondary_weights,cal_thresholds)))
                for k in range(9):profiles.append(dict(cohort=group,model=model,duration_s=duration,offset_eighths=k,eer_percent=100*metric(y,z[k])['eer']))
                for source in ('start','middle','end'):
                    base=metric(y,s[f'{duration}s_{source}_zero'])['eer']
                    for place in ('center','right'):
                        score=s[f'{duration}s_{source}_zero_{place}'];m=metric(y,score)
                        placements.append(dict(cohort=group,model=model,duration_s=duration,source_window=source,input_placement=place,
                            left_eer_percent=100*base,shifted_eer_percent=100*m['eer'],eer_shift_pp=100*(m['eer']-base),
                            mean_abs_score_change=float(np.mean(abs(score-s[f'{duration}s_{source}_zero'])))))
            print('Analyzed',group,model,'primary',result['estimate_pp'],result['interval_99_375_pp'],flush=True)
    for name,rows in [('all_cells.csv',cells),('policies.csv',policies),('input_placement.csv',placements),('offset_profiles.csv',profiles)]:write_csv(ROOT/'results'/name,rows)
    dump(ROOT/'results/expansion_summary.json',summary)


if __name__=='__main__':main()
