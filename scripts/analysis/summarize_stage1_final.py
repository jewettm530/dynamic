#!/usr/bin/env python3
# SUPERSEDED PRE-CORRECTION SUMMARY: use summarize_stage1_corrected_final.py.
"""Summarize Stage 1 final validation/test results as mean +/- sample SD."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

SEEDS=(42,2026,3407)
MODELS={
 'ef_only':('EF only','N/A'),
 'segmentation_only':('Segmentation only','N/A'),
 'multitask_W2':('Multi-task','W2'),
}
SPLITS=('val','test')


def mean_sd(x):
    a=pd.to_numeric(x,errors='coerce').dropna().to_numpy(float)
    return float(a.mean()), float(a.std(ddof=1))

def fmt(m,s,n=3): return f'{m:.{n}f} +/- {s:.{n}f}'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-root',default='output/stage1/archive_pre_correction/final_evaluation')
    ap.add_argument('--output-dir',default='results/stage1/archive_pre_correction/unmatched_comparison')
    args=ap.parse_args(); root=Path(args.input_root); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for key,(label,weight) in MODELS.items():
        for seed in SEEDS:
            for split in SPLITS:
                p=root/key/f'seed_{seed}'/f'{split}_metrics.json'
                if not p.exists(): raise FileNotFoundError(p)
                with p.open() as f: m=json.load(f)
                row={'model':label,'model_key':key,'weight':weight,'seed':seed,'split':'Validation' if split=='val' else 'Test'}
                for k in ('mae','rmse','r2','pearson_r','dice_ed','dice_es','mean_dice','mean_hd95'):
                    row[k]=m.get(k,np.nan)
                rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(out/'final_seed_results.csv',index=False)

    ef_rows=[]
    for key in ('ef_only','multitask_W2'):
        for split in ('Validation','Test'):
            g=df[(df.model_key==key)&(df.split==split)]
            row={'Model':g.model.iloc[0],'Weight':g.weight.iloc[0],'Split':split}
            for k,label in [('mae','MAE'),('rmse','RMSE'),('r2','R2'),('pearson_r','Pearson r')]:
                m,s=mean_sd(g[k]); row[label]=fmt(m,s)
            ef_rows.append(row)
    ef=pd.DataFrame(ef_rows); ef.to_csv(out/'final_ef_summary.csv',index=False)

    seg_rows=[]
    for key in ('segmentation_only','multitask_W2'):
        for split in ('Validation','Test'):
            g=df[(df.model_key==key)&(df.split==split)]
            row={'Model':g.model.iloc[0],'Weight':g.weight.iloc[0],'Split':split}
            for k,label in [('dice_ed','Dice ED'),('dice_es','Dice ES'),('mean_dice','Mean Dice'),('mean_hd95','Mean HD95')]:
                m,s=mean_sd(g[k]); row[label]=fmt(m,s)
            seg_rows.append(row)
    seg=pd.DataFrame(seg_rows); seg.to_csv(out/'final_segmentation_summary.csv',index=False)

    md=['# Stage 1 Final Results','','## EF regression','','| Model | Weight | Split | MAE ↓ | RMSE ↓ | R² ↑ | Pearson r ↑ |','|---|---|---|---:|---:|---:|---:|']
    for _,r in ef.iterrows(): md.append(f"| {r['Model']} | {r['Weight']} | {r['Split']} | {r['MAE']} | {r['RMSE']} | {r['R2']} | {r['Pearson r']} |")
    md += ['','## LV segmentation','','| Model | Weight | Split | Dice ED ↑ | Dice ES ↑ | Mean Dice ↑ | Mean HD95 ↓ |','|---|---|---|---:|---:|---:|---:|']
    for _,r in seg.iterrows(): md.append(f"| {r['Model']} | {r['Weight']} | {r['Split']} | {r['Dice ED']} | {r['Dice ES']} | {r['Mean Dice']} | {r['Mean HD95']} |")
    (out/'final_results.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))
    print(f'\nSaved final summaries to {out.resolve()}')

if __name__=='__main__': main()
