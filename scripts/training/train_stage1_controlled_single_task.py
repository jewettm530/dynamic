#!/usr/bin/env python3
# HISTORICAL T0 ONLY: two-frame controlled ablation. Use corrected B1/B2 scripts for new runs.
"""Train a controlled Stage 1 single-task ablation matching selected W2 settings."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from echonet.datasets.stage1_paired import EchoStage1PairedDataset
from echonet.modeling.stage1_controlled_baselines import EFOnlyDeepLabV3, SegmentationOnlyDeepLabV3
from echonet.utils.reproducibility import make_generator, seed_everything, seed_worker
from echonet.utils.stage1_metrics import dice_score, ef_fraction_to_percent, hd95_pixels, regression_metrics, summarize_segmentation

EXPECTED_COUNTS = {"train": 7460, "val": 1288}

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["ef","seg"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, required=True, choices=[42,2026,3407])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--regression-hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--non-deterministic", action="store_true")
    return p.parse_args()

def git_commit():
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return "unknown"

def loader(ds, a, shuffle, device):
    return DataLoader(ds,batch_size=a.batch_size,shuffle=shuffle,num_workers=a.num_workers,
        pin_memory=(device.type=="cuda"),persistent_workers=(a.num_workers>0),
        worker_init_fn=seed_worker,generator=make_generator(a.seed),drop_last=False)

def write_csv(path, rows):
    if not rows: return
    with Path(path).open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def run_ef(model, dl, device, optimizer: Optional[torch.optim.Optimizer]):
    training=optimizer is not None; model.train(training)
    criterion=torch.nn.MSELoss(reduction="mean")
    loss_sum=0.0; n=0; ys=[]; ps=[]; names=[]; started=time.time()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    with torch.set_grad_enabled(training):
        for b in dl:
            ed=b["ed_image"].to(device,non_blocking=True); es=b["es_image"].to(device,non_blocking=True)
            ef=b["ef"].to(device,non_blocking=True); bs=ef.shape[0]
            if training: optimizer.zero_grad(set_to_none=True)
            frame=model(torch.cat([ed,es],0)).reshape(-1)
            video=(frame[:bs]+frame[bs:])/2.0
            loss=criterion(video,ef)
            if training: loss.backward(); optimizer.step()
            loss_sum += float(loss.detach().cpu())*bs; n += bs
            ys.extend(ef_fraction_to_percent(ef.detach().cpu().numpy()).reshape(-1).tolist())
            ps.extend(ef_fraction_to_percent(video.detach().cpu().numpy()).reshape(-1).tolist())
            names.extend(list(b["filename"]))
    m=regression_metrics(ys,ps)
    m.update({"raw_ef_loss":loss_sum/max(n,1),"n_videos":n,"elapsed_seconds":time.time()-started,
              "peak_gpu_memory_allocated":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0})
    pred=[{"filename":f,"ef_target_percent":y,"ef_prediction_percent":p} for f,y,p in zip(names,ys,ps)]
    return m,pred

def run_seg(model, dl, device, optimizer: Optional[torch.optim.Optimizer], compute_hd95):
    training=optimizer is not None; model.train(training)
    criterion=torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss_sum=0.0; n=0; edd=[]; esd=[]; edh=[]; esh=[]; started=time.time()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    with torch.set_grad_enabled(training):
        for b in dl:
            ed=b["ed_image"].to(device,non_blocking=True); es=b["es_image"].to(device,non_blocking=True)
            edm=b["ed_mask"].to(device,non_blocking=True); esm=b["es_mask"].to(device,non_blocking=True); bs=ed.shape[0]
            images=torch.cat([ed,es],0); masks=torch.cat([edm,esm],0)
            if training: optimizer.zero_grad(set_to_none=True)
            logits=model(images); loss=criterion(logits,masks)
            if training: loss.backward(); optimizer.step()
            loss_sum += float(loss.detach().cpu())*bs; n += bs
            probs=torch.sigmoid(logits).detach().cpu().numpy()[:,0]
            truth=masks.detach().cpu().numpy()[:,0]
            for p,t in zip(probs[:bs],truth[:bs]):
                pred,true=p>=0.5,t>=0.5; edd.append(dice_score(pred,true))
                if compute_hd95: edh.append(hd95_pixels(pred,true))
            for p,t in zip(probs[bs:],truth[bs:]):
                pred,true=p>=0.5,t>=0.5; esd.append(dice_score(pred,true))
                if compute_hd95: esh.append(hd95_pixels(pred,true))
    if compute_hd95:
        m=summarize_segmentation(edd,esd,edh,esh)
    else:
        de=float(np.mean(edd)); ds=float(np.mean(esd))
        m={"dice_ed":de,"dice_es":ds,"mean_dice":(de+ds)/2.0,
           "hd95_ed":float("nan"),"hd95_es":float("nan"),"mean_hd95":float("nan")}
    m.update({"raw_seg_loss":loss_sum/max(n,1),"n_videos":n,"elapsed_seconds":time.time()-started,
              "peak_gpu_memory_allocated":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0})
    return m

def main():
    a=args(); seed_everything(a.seed,deterministic=not a.non_deterministic)
    out=Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds=EchoStage1PairedDataset(a.data_root,"train"); val_ds=EchoStage1PairedDataset(a.data_root,"val")
    actual={"train":len(train_ds),"val":len(val_ds)}
    if actual != EXPECTED_COUNTS: raise RuntimeError(f"Cohort mismatch: {actual} != {EXPECTED_COUNTS}")
    train_dl=loader(train_ds,a,True,device); val_dl=loader(val_ds,a,False,device)
    kwargs=dict(pretrained=not a.no_pretrained,regression_hidden_dim=a.regression_hidden_dim,dropout=a.dropout)
    if a.task=="ef":
        model=EFOnlyDeepLabV3(**kwargs).to(device); rule="lowest validation EF MAE"; loss_name="MSELoss(mean)"
    else:
        model=SegmentationOnlyDeepLabV3(**kwargs).to(device); rule="highest validation Mean Dice"; loss_name="BCEWithLogitsLoss(mean)"
    optimizer=Adam(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
    config=vars(a).copy()
    config.update({"reference_multitask":"W2","architecture_family":"MultitaskDeepLabV3 controlled ablation",
        "shared_backbone":"DeepLabV3-ResNet50","input":"paired labeled ED/ES frames","train_videos":len(train_ds),
        "validation_videos":len(val_ds),"optimizer":"Adam","loss":loss_name,"checkpoint_rule":rule,
        "spatial_augmentation":"none","ef_training_target_scale":"0-1 fraction",
        "ef_evaluation_scale":"0-100 percentage points","git_commit":git_commit(),"command":" ".join(sys.argv)})
    with (out/"run_config.json").open("w") as f: json.dump(config,f,indent=2)
    history=[]; best=float("inf") if a.task=="ef" else -float("inf"); best_epoch=None
    for epoch in range(1,a.epochs+1):
        if a.task=="ef":
            tr,_=run_ef(model,train_dl,device,optimizer); va,preds=run_ef(model,val_dl,device,None)
            value=va["mae"]; better=value<best
        else:
            tr=run_seg(model,train_dl,device,optimizer,False); va=run_seg(model,val_dl,device,None,True); preds=None
            value=va["mean_dice"]; better=value>best
        for phase,m in [("train",tr),("val",va)]:
            row={"epoch":epoch,"phase":phase,"seed":a.seed}; row.update(m); history.append(row)
        write_csv(out/"training_history.csv",history)
        ck={"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
            "val_metrics":va,"config":config}
        torch.save(ck,out/"checkpoint.pt")
        if better:
            best=value; best_epoch=epoch; torch.save(ck,out/"best.pt")
            if preds is not None: write_csv(out/"best_validation_predictions.csv",preds)
            with (out/"best_validation_metrics.json").open("w") as f: json.dump(va,f,indent=2)
        if a.task=="ef":
            print(f"Epoch {epoch:03d} | val MAE={va['mae']:.3f} RMSE={va['rmse']:.3f} R2={va['r2']:.3f} r={va['pearson_r']:.3f}")
        else:
            print(f"Epoch {epoch:03d} | Dice ED={va['dice_ed']:.4f} ES={va['dice_es']:.4f} Mean={va['mean_dice']:.4f} HD95={va['mean_hd95']:.3f}")
    summary={"task":a.task,"best_epoch":best_epoch,"checkpoint_rule":rule,"best_validation_value":best,
             "best_checkpoint":str(out/"best.pt")}
    with (out/"run_summary.json").open("w") as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
