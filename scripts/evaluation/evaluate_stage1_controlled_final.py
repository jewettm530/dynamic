#!/usr/bin/env python3
"""Evaluate controlled EF-only, segmentation-only, and locked W2 on val/test."""
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from echonet.datasets.stage1_paired import EchoStage1PairedDataset
from echonet.modeling.multitask_deeplab import MultitaskDeepLabV3
from echonet.modeling.stage1_controlled_baselines import EFOnlyDeepLabV3, SegmentationOnlyDeepLabV3
from echonet.utils.reproducibility import make_generator, seed_everything, seed_worker
from echonet.utils.stage1_metrics import dice_score, ef_fraction_to_percent, hd95_pixels, regression_metrics, summarize_segmentation

SEEDS=(42,2026,3407); EXPECTED={"val":1288,"test":1276}; EVAL_SEED=8675309

def dl(ds,device,batch=4,workers=4):
    return DataLoader(ds,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=True,
        persistent_workers=(workers>0),worker_init_fn=seed_worker,generator=make_generator(EVAL_SEED),drop_last=False)

def load(path,device): return torch.load(path,map_location=device,weights_only=False)

def eval_ef(model,loader,device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for b in loader:
            ed=b["ed_image"].to(device); es=b["es_image"].to(device); ef=b["ef"].to(device); n=ef.shape[0]
            frame=model(torch.cat([ed,es],0)).reshape(-1); video=(frame[:n]+frame[n:])/2
            ys += ef_fraction_to_percent(ef.cpu().numpy()).reshape(-1).tolist()
            ps += ef_fraction_to_percent(video.cpu().numpy()).reshape(-1).tolist()
    return regression_metrics(ys,ps)

def seg_metrics_from_logits(logits,masks,n,edd,esd,edh,esh):
    probs=torch.sigmoid(logits).cpu().numpy()[:,0]; truth=masks.cpu().numpy()[:,0]
    for p,t in zip(probs[:n],truth[:n]):
        pred,true=p>=.5,t>=.5; edd.append(dice_score(pred,true)); edh.append(hd95_pixels(pred,true))
    for p,t in zip(probs[n:],truth[n:]):
        pred,true=p>=.5,t>=.5; esd.append(dice_score(pred,true)); esh.append(hd95_pixels(pred,true))

def eval_seg(model,loader,device):
    model.eval(); edd=[]; esd=[]; edh=[]; esh=[]
    with torch.no_grad():
        for b in loader:
            ed=b["ed_image"].to(device); es=b["es_image"].to(device)
            masks=torch.cat([b["ed_mask"].to(device),b["es_mask"].to(device)],0); n=ed.shape[0]
            seg_metrics_from_logits(model(torch.cat([ed,es],0)),masks,n,edd,esd,edh,esh)
    return summarize_segmentation(edd,esd,edh,esh)

def eval_mtl(model,loader,device):
    model.eval(); ys=[]; ps=[]; edd=[]; esd=[]; edh=[]; esh=[]
    with torch.no_grad():
        for b in loader:
            ed=b["ed_image"].to(device); es=b["es_image"].to(device); ef=b["ef"].to(device); n=ef.shape[0]
            masks=torch.cat([b["ed_mask"].to(device),b["es_mask"].to(device)],0)
            out=model(torch.cat([ed,es],0)); frame=out["ef"].reshape(-1); video=(frame[:n]+frame[n:])/2
            ys += ef_fraction_to_percent(ef.cpu().numpy()).reshape(-1).tolist()
            ps += ef_fraction_to_percent(video.cpu().numpy()).reshape(-1).tolist()
            seg_metrics_from_logits(out["segmentation"],masks,n,edd,esd,edh,esh)
    m=regression_metrics(ys,ps); m.update(summarize_segmentation(edd,esd,edh,esh)); return m

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",default="/data/jewettm/dynamic/datasets")
    p.add_argument("--controlled-root",default="output/stage1/final_baselines_controlled")
    p.add_argument("--mtl-root",default="output/stage1/weighting/W2")
    p.add_argument("--output-root",default="output/stage1/final_evaluation_controlled")
    a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=Path(a.output_root); out.mkdir(parents=True,exist_ok=True)
    for split in ("val","test"):
        ds=EchoStage1PairedDataset(a.data_root,split)
        if len(ds)!=EXPECTED[split]: raise RuntimeError(f"{split}: {len(ds)} != {EXPECTED[split]}")
        for seed in SEEDS:
            seed_everything(EVAL_SEED,True); loader=dl(ds,device)
            ck=load(Path(a.controlled_root)/"ef_only"/f"seed_{seed}"/"best.pt",device)
            cfg=ck["config"]; m=EFOnlyDeepLabV3(False,int(cfg["regression_hidden_dim"]),float(cfg["dropout"])).to(device)
            m.load_state_dict(ck["model_state_dict"]); metrics=eval_ef(m,loader,device); del m; torch.cuda.empty_cache()
            d=out/"ef_only"/f"seed_{seed}"; d.mkdir(parents=True,exist_ok=True)
            json.dump(metrics,(d/f"{split}_metrics.json").open("w"),indent=2)

            seed_everything(EVAL_SEED,True); loader=dl(ds,device)
            ck=load(Path(a.controlled_root)/"segmentation_only"/f"seed_{seed}"/"best.pt",device)
            cfg=ck["config"]; m=SegmentationOnlyDeepLabV3(False,int(cfg["regression_hidden_dim"]),float(cfg["dropout"])).to(device)
            m.load_state_dict(ck["model_state_dict"]); metrics=eval_seg(m,loader,device); del m; torch.cuda.empty_cache()
            d=out/"segmentation_only"/f"seed_{seed}"; d.mkdir(parents=True,exist_ok=True)
            json.dump(metrics,(d/f"{split}_metrics.json").open("w"),indent=2)

            seed_everything(EVAL_SEED,True); loader=dl(ds,device)
            ck=load(Path(a.mtl_root)/f"seed_{seed}"/"best.pt",device); cfg=ck["config"]
            m=MultitaskDeepLabV3(False,int(cfg.get("regression_hidden_dim",256)),float(cfg.get("dropout",.3))).to(device)
            m.load_state_dict(ck["model_state_dict"]); metrics=eval_mtl(m,loader,device); del m; torch.cuda.empty_cache()
            d=out/"multitask_W2"/f"seed_{seed}"; d.mkdir(parents=True,exist_ok=True)
            json.dump(metrics,(d/f"{split}_metrics.json").open("w"),indent=2)
        print("Completed:",split)
    print("Evaluation complete:",out.resolve())

if __name__=="__main__": main()
