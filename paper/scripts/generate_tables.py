#!/usr/bin/env python3
"""Render all current paper numbers from the checksum-bound recovery registry."""
from pathlib import Path
from recovery_data import ROOT, REGISTRY as R, select, value


def table(name, header, rows):
    content = '\\begin{tabular}{' + 'l' + 'r' * (len(header)-1) + '}\n\\toprule\n'
    content += ' & '.join(header) + ' \\\\\n\\midrule\n'
    content += ''.join(' & '.join(map(str, row)) + ' \\\\\n' for row in rows)
    content += '\\bottomrule\n\\end{tabular}\n'
    (ROOT / 'tables' / name).write_text(content)


def main():
    (ROOT / 'tables').mkdir(exist_ok=True)
    macros = {}
    rows = []
    for ds in ('r42', 'r62'):
        p = R['populations'][ds]
        rows.append([ds.replace('r4', 'r4.').replace('r6', 'r6.'), f"{p['rows']:,}",
                     f"{p['my_positive_days']:,}", p['my_positive_users'], p['changed_rows']])
        for key in ('rows','my_positive_days','my_positive_users','changed_rows'):
            macros[f'{ds}.{key}'] = f'{p[key]:,}'
        for pop in ('all_users', 'unseen_users'):
            for metric in ('day_roc', 'day_ap', 'user_roc'):
                v = value(ds+'_main', 'adapted_nll', metric, pop)
                macros[f'{ds}.{pop}.{metric}'] = f'{v:.3f}' if metric != 'day_ap' else f'{v:.6f}'
    table('recovery_populations.tex', ['Dataset','Observed days','Positive days','Insiders','Changed labels'], rows)
    rows = []
    for ds in ('r42', 'r62'):
        for pop in ('all_users', 'unseen_users'):
            rows.append([('r4.2' if ds=='r42' else 'r6.2')+' / '+('general' if pop=='all_users' else 'unseen'),
                         *[macros[f'{ds}.{pop}.{m}'] for m in ('day_roc','day_ap','user_roc')]])
    table('recovery_detector.tex', ['Dataset / negatives','Day ROC','Day AP','User ROC'], rows)
    rows=[]
    for view,label in [('full','Full score'),('behavior_only','Behavior targets'),('profile_only','Profile targets')]:
        vals=R['c32']['fold_mean'][view]
        rows.append([label,f"{vals['day_roc']:.3f}",f"{vals['day_ap']:.6f}",f"{vals['user_roc']:.3f}"])
        for m in ('day_roc','day_ap','user_roc'):
            macros[f'c32.{view}.{m}']=f'{vals[m]:.6f}' if m=='day_ap' else f'{vals[m]:.3f}'
    table('recovery_behavior.tex',['r4.2 score','Day ROC','Day AP','User ROC'],rows)
    gain=R['c32']['behavior_minus_full_day_roc']
    macros['c32.gain']=f"{gain['estimate']:.3f}"
    macros['c32.gain.ci']=f"[{gain['ci95'][0]:.3f}, {gain['ci95'][1]:.3f}]"
    macros['c32.fold.prevalence']=f"{R['c32']['fold_mean']['full']['prevalence']:.6f}"
    macros['c32.pooled.prevalence']=f"{R['c32']['pooled_prevalence']:.6f}"
    rows=[]
    for scale in ('3b','8b'):
        for view,label in [('full','Full'),('no_psy','No personality'),('no_profile','No profile'),('shuffle_profile','Shuffled profile')]:
            run='r62_factorial_'+scale
            vals=[value(run,view,m) for m in ('day_roc','user_roc')]
            rows.append([scale.upper()+' / '+label,*[f'{v:.3f}' for v in vals]])
            for m,v in zip(('day_roc','user_roc'),vals):macros[f'factorial.{scale}.{view}.{m}']=f'{v:.3f}'
    table('recovery_factorial.tex',['r6.2 adapter / input','Day ROC','User ROC'],rows)
    rows=[]
    for scale,run in [('3B full training','r62_classes_3b_full'),('3B target-mask training','r62_classes_3b_targetmask'),('8B full training','r62_classes_8b_full')]:
        rows.append([scale,f"{value(run,'full'):.3f}",f"{value(run,'behavior_only'):.3f}",f"{value(run,'behavior_only','day_ap'):.6f}"])
    table('recovery_r62_behavior.tex',['Separate r6.2 adapters','Full ROC','Behavior ROC','Behavior AP'],rows)
    rows=[]
    for suffix,label in [('w1','One observation'),('w7','Seven incl. target'),('fc7','Forecast (past only)')]:
        row=select('pooled',run='r62_timeseries_'+suffix,population='common_history_unseen',view='full')
        rows.append([label,*[f"{row[m]:.6f}" if m=='day_ap' else f"{row[m]:.3f}" for m in ('day_roc','day_ap','user_roc')]])
    table('recovery_timeseries.tex',['r6.2 numeric model','Pooled day ROC','Pooled AP','User ROC'],rows)
    rows=[]
    for endpoint,label in [('primary rpU_S - randI_S (attack)','Selected minus isotropic'),('co-primary rpU_S - rpU_C (attack)','Selected minus active'),('specificity [rpU_S - rpU_C](attack) - (window-negative)','Attack specificity')]:
        row=R['pilot_diagnostic']['endpoints'][endpoint]
        rows.append([label,f"{row['estimate']:+.6f}",f"[{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}]"])
    table('recovery_diagnostic.tex',['Post-run diagnostic','Effect','95\\% interval'],rows)
    for c in R['claims']:
        macros['claim.'+c['id']]=c['current_disposition'].replace('_', ' ')
    (ROOT/'tables/recovery_values.tex').write_text('% Generated; do not edit.\n'+''.join(
        '\\expandafter\\def\\csname rv@'+k+'\\endcsname{'+v+'}\n' for k,v in sorted(macros.items())))
    print('Generated seven scoped result tables and shared values.')


if __name__=='__main__':main()
