#!/usr/bin/env python3
"""Generate the current shared score-mitigation figure from the registry."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recovery_data import ROOT, REGISTRY as R


def main():
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(6.1,2.5))
    for ax,metric,title in zip(axes,('day_roc','day_ap'),('Mean fold day ROC-AUC','Mean fold day AP')):
        vals=[R['c32']['fold_mean'][v][metric] for v in ('full','behavior_only','profile_only')]
        ax.barh(['Full score','Behavior targets','Profile targets'],vals,color=['#738296','#2f5fa5','#b6bdc5'])
        ax.invert_yaxis();ax.set_title(title,loc='left',fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(0,1 if metric=='day_roc' else max(vals)*1.27)
        for i,v in enumerate(vals):ax.text(v+.01 if metric=='day_roc' else v+.0005,i,f'{v:.3f}' if metric=='day_roc' else f'{v:.4f}',va='center',fontsize=8)
    fig.suptitle('r4.2: same model and inputs; only scored targets change',fontsize=10)
    fig.tight_layout();(ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/recovery_score_mitigation.pdf')
    fig.savefig(ROOT/'figures/recovery_score_mitigation.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
