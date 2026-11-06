"""Standalone Matplotlib plots from the three adjoining figure CSV files."""
import argparse
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=('aasist','ssl','rawnet2','antideepfake')
LABELS={'aasist':'AASIST','ssl':'SSL-AASIST','rawnet2':'RawNet2','antideepfake':'AntiDF-S'}


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def save(fig,path):
    for suffix in ('pdf','svg','png'):fig.savefig(path.with_suffix('.'+suffix),dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args();out=args.data_dir
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,'axes.labelsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    primary=read(out/'figure_primary.csv');placement=read(out/'figure_placement.csv');profiles=read(out/'figure_profiles.csv')
    fig,axes=plt.subplots(2,1,figsize=(4.8,3.6),sharex=True,layout='constrained')
    for ax,group,title,color in zip(axes,('vi','itw'),('a  Fresh Vietnamese files','b  Fresh In-the-Wild files'),('#146a8d','#a14a24')):
        for j,model in enumerate(MODELS):
            row=next(r for r in primary if r['cohort']==group and r['model']==model)
            ax.hlines(j,float(row['lo']),float(row['hi']),color=color,lw=1.6)
            ax.scatter(float(row['estimate_pp']),j,color=color,s=26,zorder=3)
        ax.axvline(0,color='.6',lw=.8,ls='--');ax.set_yticks(range(4),[LABELS[m] for m in MODELS]);ax.set_ylim(3.5,-.5)
        ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Middle minus start EER (percentage points)')
    save(fig,out/'primary_replication')
    fig,axes=plt.subplots(2,1,figsize=(4.8,3.8),sharex=True,layout='constrained')
    for ax,group,title in zip(axes,('vi','itw'),('a  Fresh Vietnamese files','b  Fresh In-the-Wild files')):
        for j,model in enumerate(MODELS):
            for offset,name,color,marker in ((-.12,'center','#146a8d','o'),(.12,'right','#a14a24','s')):
                row=next(r for r in placement if r['cohort']==group and r['model']==model and r['placement']==name)
                ax.hlines(j+offset,float(row['lo']),float(row['hi']),color=color,lw=1.3)
                ax.scatter(float(row['estimate_pp']),j+offset,color=color,s=20,marker=marker,label=name if j==0 else None)
        ax.axvline(0,color='.6',lw=.8,ls='--');ax.set_yticks(range(4),[LABELS[m] for m in MODELS]);ax.set_ylim(3.5,-.5)
        ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('EER change from left placement (points)')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.58,-.015),ncol=2,frameon=False)
    save(fig,out/'identical_content_placement')
    fig,axes=plt.subplots(2,2,figsize=(4.8,3.5),sharex=True,sharey=True,layout='constrained')
    maximum=max(65,5*math.ceil(max(float(r['eer_percent']) for r in profiles)/5))
    for ax,model in zip(axes.flat,MODELS):
        for group,color in (('vi','#146a8d'),('itw','#a14a24')):
            for duration,style in ((1,'-'),(2,'--')):
                rows=sorted((r for r in profiles if r['cohort']==group and r['model']==model and int(r['duration_s'])==duration),key=lambda r:int(r['offset_eighths']))
                ax.plot([int(r['offset_eighths'])/8 for r in rows],[float(r['eer_percent']) for r in rows],ls=style,color=color,lw=1.2,marker='o',ms=2.3,label=f'{group.upper()}, {duration}s')
        ax.set_title(LABELS[model],loc='left',fontweight='bold');ax.set_xticks([0,.5,1]);ax.set_ylim(0,maximum)
    for ax in axes[:,0]:ax.set_ylabel('EER (%)')
    for ax in axes[-1,:]:ax.set_xlabel('Relative source offset')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False,fontsize=10,columnspacing=1,handlelength=1.4)
    save(fig,out/'offset_profiles')


if __name__=='__main__':main()
