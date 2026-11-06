import collections
import datetime
import hashlib
from pathlib import Path
import shutil
from common import *


def main():
    assert not (ROOT/'selection_plan.json').exists(),'Selection already fixed'
    prior=list((CORE/'runs/shortform_pilot_20261106').glob('*manifest.csv'))
    prior += [CORE/'holdout_vi_20261106/candidate_manifest.csv',CORE/'external_itw_20261106/candidate_manifest.csv']
    excluded={r['utterance_id'] for p in prior for r in read(p) if 'utterance_id' in r}
    meta=PROJECT/'feasibility_20261106'
    languages=dict(s.split() for s in (meta/'lang_RADAR2026-eval.txt').read_text().splitlines())
    pools={'bonafide':[],'spoof':[]}
    for line in (meta/'label_RADAR2026-eval.txt').read_text().splitlines():
        a=line.split()
        if a[0] in excluded or languages[a[0]]!='VI' or float(a[1])<4.04:continue
        pools[a[2]].append(dict(utterance_id=a[0],language='VI',label=a[2],metadata_duration_s=a[1],segments=' '.join(a[3:])))
    candidates=[]
    for label,items in pools.items():
        items.sort(key=lambda r:hashlib.sha256(('expanded-vi-20261106:'+r['utterance_id']).encode()).hexdigest())
        assert len(items)>=700
        candidates.extend(items[:700])
    write_csv(ROOT/'vi/candidate_manifest.csv',candidates)
    vi_counts={k:len(v) for k,v in pools.items()}
    source=CORE/'external_itw_20261106/source'
    index={Path(r['filename']).name:r for r in read(source/'zip_member_index.csv')}
    probes={r['file'] for r in load(source/'format_probe.json')}
    pools={'bonafide':[],'spoof':[]}
    for row in read(source/'meta.csv'):
        name=row['file'];uid='ITW_'+Path(name).stem;member=index[name]
        duration=(int(member['file_size'])-58)/64000
        if uid in excluded or name in probes or duration<4.04:continue
        label='bonafide' if row['label']=='bona-fide' else row['label']
        assert label in pools
        pools[label].append(dict(utterance_id=uid,language='EN',label=label,file=name,metadata_duration_s=duration,
            speaker_group=hashlib.sha256(('itw-speaker:'+row['speaker']).encode()).hexdigest()[:20],
            **{k:member[k] for k in ('filename','file_size','compress_size','CRC','header_offset','compress_type')}))
    candidates=[]
    for label,items in pools.items():
        items.sort(key=lambda r:hashlib.sha256(('expanded-itw-20261106:'+r['utterance_id']).encode()).hexdigest())
        assert len(items)>=1100
        candidates.extend(items[:1100])
    write_csv(ROOT/'itw/candidate_manifest.csv',candidates)
    itw=ROOT/'itw';(itw/'source').mkdir(exist_ok=True)
    shutil.copy2(source/'hf_repository.json',itw/'source/hf_repository.json')
    old=load(CORE/'external_itw_20261106/external_plan.json')
    dump(itw/'external_plan.json',dict(source=old['source'],upstream_commit=old['upstream_commit'],candidate_manifest_sha256=sha(itw/'candidate_manifest.csv')))
    dump(ROOT/'selection_plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        vi_eligible=vi_counts,itw_eligible={k:len(v) for k,v in pools.items()},targets={'vi':{'bonafide':600,'spoof':600},'itw':{'bonafide':1000,'spoof':1000}},
        prior_manifest_sha256={str(p.relative_to(PROJECT)):sha(p) for p in prior},
        candidate_sha256={g:sha(ROOT/g/'candidate_manifest.csv') for g in ('vi','itw')},
        selection='Within-class SHA256 ranking, all previously considered candidates excluded; only data-quality reserve substitutions; no outcomes viewed',
        pcm_audit_required=True))
    print('Fresh candidate selection fixed; VI',vi_counts,'ITW',{k:len(v) for k,v in pools.items()},flush=True)


if __name__=='__main__':main()
