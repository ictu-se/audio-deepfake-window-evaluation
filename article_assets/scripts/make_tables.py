"""Rebuild expanded-manuscript tables from the accompanying frozen result copies."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
MODELS=('aasist','ssl','rawnet2','antideepfake')
NAMES={'aasist':'AASIST','ssl':'SSL-AASIST','rawnet2':'RawNet2','antideepfake':'AntiDF-S'}


def read(name):
    with (DATA/name).open() as stream:return list(csv.DictReader(stream))


def table(name,columns,heading,rows):
    text='\\begin{tabular}{@{}'+columns+'@{}}\n\\toprule\n'+heading+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(row)+r' \\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n';(DATA/name).write_text(text)


def main():
    summary=json.loads((DATA/'expansion_summary.json').read_text())
    cells={(r['cohort'],r['model'],r['condition']):r for r in read('all_cells.csv')}
    primary=[]
    for group in ('vi','itw'):
        for i,model in enumerate(MODELS):
            key=group+'_'+model;r=summary['primary'][key];ref=cells[group,model,'official'];lo,hi=r['interval_99_375_pp']
            primary.append([group.upper() if i==0 else '',NAMES[model],f"{float(ref['eer_percent']):.1f}",f"{float(ref['auc']):.3f}",f"{r['estimate_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    table('tab_primary.tex','llrrr',r'Cohort & Pipeline & Ref. EER & Ref. AUC & $\Delta_{\rm source}$ [CI]',primary)
    policies=read('policies.csv');rows=[]
    for group in ('vi','itw'):
        for i,model in enumerate(MODELS):
            vals={policy:np.mean([float(r['eer_percent']) for r in policies if r['cohort']==group and r['model']==model and r['policy']==policy]) for policy in ('start','middle','random','mean_three','vad_max')}
            rows.append([group.upper() if i==0 else '',NAMES[model]]+[f'{v:.2f}' for v in vals.values()])
    table('tab_policies.tex','llrrrrr','Cohort & Pipeline & Start & Middle & Random & Mean3 & VAD',rows)
    operating=[]
    for model in MODELS:
        for i,policy in enumerate(('start','mean_three')):
            candidates=[r for r in policies if r['cohort']=='itw' and r['model']==model and int(r['duration_s'])==1 and r['policy']==policy]
            assert len(candidates)==1
            r=candidates[0]
            operating.append([NAMES[model] if i==0 else '', 'Start' if policy=='start' else 'Mean3']+
                [f"{float(r[point]):.1f} [{float(r[lo]):.1f}, {float(r[hi]):.1f}]" for point,lo,hi in
                 [('bonafide_block_percent','bonafide_block_lo95','bonafide_block_hi95'),('spoof_pass_percent','spoof_pass_lo95','spoof_pass_hi95')]])
    table('tab_operating.tex','llrr','Pipeline & Policy & Bona fide blocked & Spoofs passed',operating)
    runtime=read('policy_runtime.csv');rows=[]
    for model in MODELS:
        row=[NAMES[model]]
        for policy in ('start','mean_three','vad_max'):
            values=[float(r['total_seconds'])*1000 for r in runtime if r['model']==model and r['policy']==policy]
            assert len(values)==48;row.append(f'{np.median(values):.1f}')
        rows.append(row)
    table('tab_runtime.tex','lrrr','Pipeline & Start & Mean3 & VAD selection',rows)
    print('Four expanded manuscript tables generated from frozen result files.')


if __name__=='__main__':main()
