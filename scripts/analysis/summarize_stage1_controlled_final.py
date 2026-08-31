#!/usr/bin/env python3
# HISTORICAL T0 ONLY: use summarize_stage1_corrected_final.py.
import json
from pathlib import Path
import numpy as np, pandas as pd
SEEDS=(42,2026,3407); SPLITS=("val","test")
MODELS={"ef_only":("EF only","N/A"),"segmentation_only":("Segmentation only","N/A"),"multitask_W2":("Multi-task","W2")}
root=Path("output/stage1/T0_two_frame_oracle/final_evaluation_controlled"); out=Path("results/stage1/T0_two_frame_oracle/final_controlled"); out.mkdir(parents=True,exist_ok=True)
rows=[]
for key,(label,weight) in MODELS.items():
    for seed in SEEDS:
        for split in SPLITS:
            p=root/key/f"seed_{seed}"/f"{split}_metrics.json"
            m=json.load(p.open())
            r={"model":label,"key":key,"weight":weight,"seed":seed,"split":"Validation" if split=="val" else "Test"}
            for x in ("mae","rmse","r2","pearson_r","dice_ed","dice_es","mean_dice","mean_hd95"): r[x]=m.get(x,np.nan)
            rows.append(r)
df=pd.DataFrame(rows); df.to_csv(out/"final_seed_results.csv",index=False)

def ms(g,col):
    x=pd.to_numeric(g[col],errors="coerce").dropna().to_numpy(float)
    if len(x)!=3: raise ValueError(f"{col}: expected 3 values, got {len(x)}")
    return f"{x.mean():.3f} +/- {x.std(ddof=1):.3f}"

ef=[]
for key in ("ef_only","multitask_W2"):
    for split in ("Validation","Test"):
        g=df[(df.key==key)&(df.split==split)]
        ef.append({"Model":g.model.iloc[0],"Weight":g.weight.iloc[0],"Split":split,
                   "MAE":ms(g,"mae"),"RMSE":ms(g,"rmse"),"R2":ms(g,"r2"),"Pearson r":ms(g,"pearson_r")})
ef=pd.DataFrame(ef); ef.to_csv(out/"final_ef_summary.csv",index=False)
seg=[]
for key in ("segmentation_only","multitask_W2"):
    for split in ("Validation","Test"):
        g=df[(df.key==key)&(df.split==split)]
        seg.append({"Model":g.model.iloc[0],"Weight":g.weight.iloc[0],"Split":split,
                    "Dice ED":ms(g,"dice_ed"),"Dice ES":ms(g,"dice_es"),"Mean Dice":ms(g,"mean_dice"),"Mean HD95":ms(g,"mean_hd95")})
seg=pd.DataFrame(seg); seg.to_csv(out/"final_segmentation_summary.csv",index=False)

lines=["# Stage 1 Final Results — Controlled Comparison","","## EF regression","",
"| Model | Weight | Split | MAE ↓ | RMSE ↓ | R² ↑ | Pearson r ↑ |","|---|---|---|---:|---:|---:|---:|"]
for _,r in ef.iterrows(): lines.append(f"| {r['Model']} | {r['Weight']} | {r['Split']} | {r['MAE']} | {r['RMSE']} | {r['R2']} | {r['Pearson r']} |")
lines += ["","## LV segmentation","",
"| Model | Weight | Split | Dice ED ↑ | Dice ES ↑ | Mean Dice ↑ | Mean HD95 ↓ |","|---|---|---|---:|---:|---:|---:|"]
for _,r in seg.iterrows(): lines.append(f"| {r['Model']} | {r['Weight']} | {r['Split']} | {r['Dice ED']} | {r['Dice ES']} | {r['Mean Dice']} | {r['Mean HD95']} |")
text="\n".join(lines)+"\n"; (out/"final_results.md").write_text(text); print(text)
