#!/usr/bin/env python3
"""Evaluate Stage 1 final checkpoints on validation and test.

This script is run only AFTER W2 is locked. It evaluates:
  * EF-only: 3 selected checkpoints
  * segmentation-only: 3 selected checkpoints
  * selected multi-task W2: the 3 Step 2 selected checkpoints

No checkpoint or hyperparameter is selected using test results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50

from echonet.datasets.echo import Echo
from echonet.modeling.multitask_deeplab import MultitaskDeepLabV3
from echonet.utils.reproducibility import make_generator, seed_everything, seed_worker
from echonet.utils.stage1_metrics import (
    dice_score, ef_fraction_to_percent, hd95_pixels, regression_metrics,
    summarize_segmentation,
)

SEEDS = (42, 2026, 3407)
EXPECTED = {"val": 1288, "test": 1276}
EVAL_SEED = 8675309  # fixed across model seeds so EF temporal sampling is identical


class EchoEFCommon(Dataset):
    def __init__(self, root, split, frames=32, period=2):
        self.ds = Echo(
            root=root, split=split,
            target_type=["Filename", "EF", "LargeIndex", "SmallIndex"],
            mean=np.array([0.,0.,0.]), std=np.array([1.,1.,1.]),
            length=frames, period=period, clips=1, pad=None, noise=None,
        )
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        video, target = self.ds[i]
        filename, ef, _, _ = target
        return torch.as_tensor(video,dtype=torch.float32), torch.tensor(float(ef),dtype=torch.float32), filename


class EchoSeg(Dataset):
    def __init__(self, root, split, include_ef=False):
        targets=["Filename","LargeFrame","SmallFrame","LargeTrace","SmallTrace"]
        if include_ef: targets.append("EF")
        self.include_ef=include_ef
        self.ds=Echo(root=root,split=split,target_type=targets,mean=0.0,std=1.0,
                     length=16,period=2,clips=1,pad=None,noise=None)
    def __len__(self): return len(self.ds)
    def __getitem__(self,i):
        _, t=self.ds[i]
        if self.include_ef:
            filename,ed,es,edm,esm,ef=t
        else:
            filename,ed,es,edm,esm=t; ef=None
        out={
            'filename':filename,
            'ed_image':torch.as_tensor(ed,dtype=torch.float32),
            'es_image':torch.as_tensor(es,dtype=torch.float32),
            'ed_mask':(torch.as_tensor(edm,dtype=torch.float32).unsqueeze(0)>0.5).float(),
            'es_mask':(torch.as_tensor(esm,dtype=torch.float32).unsqueeze(0)>0.5).float(),
        }
        if ef is not None: out['ef']=torch.tensor(float(ef)/100.0,dtype=torch.float32)
        return out


def loader(ds,batch_size,num_workers,device):
    return DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,
                      pin_memory=(device.type=='cuda'),persistent_workers=(num_workers>0),
                      worker_init_fn=seed_worker,generator=make_generator(EVAL_SEED),drop_last=False)


def build_ef():
    model=torchvision.models.video.r2plus1d_18(weights=None)
    model.fc=torch.nn.Linear(model.fc.in_features,1)
    return model


def build_seg():
    model=deeplabv3_resnet50(weights=None,weights_backbone=None,aux_loss=False)
    last=model.classifier[-1]
    model.classifier[-1]=torch.nn.Conv2d(last.in_channels,1,kernel_size=1)
    return model


def load_state(model,path,device):
    ckpt=torch.load(path,map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return ckpt


def eval_ef(model,dl,device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y,_ in dl:
            x=x.to(device,non_blocking=True); y=y.numpy().reshape(-1)
            pred=model(x).reshape(-1).cpu().numpy()
            ys.extend(y.tolist()); ps.extend(ef_fraction_to_percent(pred).tolist())
    return regression_metrics(ys,ps)


def eval_seg(model,dl,device):
    model.eval(); edd=[]; esd=[]; edh=[]; esh=[]
    with torch.no_grad():
        for b in dl:
            ed=b['ed_image'].to(device); es=b['es_image'].to(device)
            edm=b['ed_mask'].to(device); esm=b['es_mask'].to(device); n=ed.shape[0]
            logits=model(torch.cat([ed,es],0))['out']
            probs=torch.sigmoid(logits).cpu().numpy()[:,0]
            truth=torch.cat([edm,esm],0).cpu().numpy()[:,0]
            for p,t in zip(probs[:n],truth[:n]):
                pred=p>=0.5; true=t>=0.5; edd.append(dice_score(pred,true)); edh.append(hd95_pixels(pred,true))
            for p,t in zip(probs[n:],truth[n:]):
                pred=p>=0.5; true=t>=0.5; esd.append(dice_score(pred,true)); esh.append(hd95_pixels(pred,true))
    return summarize_segmentation(edd,esd,edh,esh)


def eval_mtl(model,dl,device):
    model.eval(); ys=[]; ps=[]; edd=[]; esd=[]; edh=[]; esh=[]
    with torch.no_grad():
        for b in dl:
            ed=b['ed_image'].to(device); es=b['es_image'].to(device)
            edm=b['ed_mask'].to(device); esm=b['es_mask'].to(device); ef=b['ef'].to(device); n=ed.shape[0]
            out=model(torch.cat([ed,es],0)); frame=out['ef'].reshape(-1)
            video=(frame[:n]+frame[n:])/2.0
            ys.extend(ef_fraction_to_percent(ef.cpu().numpy()).reshape(-1).tolist())
            ps.extend(ef_fraction_to_percent(video.cpu().numpy()).reshape(-1).tolist())
            probs=torch.sigmoid(out['segmentation']).cpu().numpy()[:,0]
            truth=torch.cat([edm,esm],0).cpu().numpy()[:,0]
            for p,t in zip(probs[:n],truth[:n]):
                pred=p>=0.5; true=t>=0.5; edd.append(dice_score(pred,true)); edh.append(hd95_pixels(pred,true))
            for p,t in zip(probs[n:],truth[n:]):
                pred=p>=0.5; true=t>=0.5; esd.append(dice_score(pred,true)); esh.append(hd95_pixels(pred,true))
    m=regression_metrics(ys,ps); m.update(summarize_segmentation(edd,esd,edh,esh)); return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',default='/data/jewettm/dynamic/datasets')
    ap.add_argument('--ef-root',default='output/stage1/final_baselines/ef_only')
    ap.add_argument('--seg-root',default='output/stage1/final_baselines/segmentation_only')
    ap.add_argument('--mtl-root',default='output/stage1/weighting/W2')
    ap.add_argument('--output-root',default='output/stage1/final_evaluation')
    ap.add_argument('--batch-size-ef',type=int,default=20)
    ap.add_argument('--batch-size-seg',type=int,default=20)
    ap.add_argument('--batch-size-mtl',type=int,default=4)
    ap.add_argument('--num-workers',type=int,default=4)
    args=ap.parse_args()

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise SystemExit('ERROR: CUDA required for final evaluation')
    outroot=Path(args.output_root); outroot.mkdir(parents=True,exist_ok=True)

    manifest={'locked_weight':'W2','test_allowed':True,'test_used_for_selection':False,
              'fixed_evaluation_seed':EVAL_SEED,'expected_counts':EXPECTED}
    with (outroot/'evaluation_manifest.json').open('w') as f: json.dump(manifest,f,indent=2)

    for split in ('val','test'):
        # Construct once per split; fixed evaluation generator gives identical EF clip sampling across model seeds.
        ef_ds=EchoEFCommon(args.data_root,split)
        seg_ds=EchoSeg(args.data_root,split,False)
        mtl_ds=EchoSeg(args.data_root,split,True)
        if not (len(ef_ds)==len(seg_ds)==len(mtl_ds)==EXPECTED[split]):
            raise RuntimeError(f'{split} cohort mismatch: EF={len(ef_ds)}, seg={len(seg_ds)}, mtl={len(mtl_ds)}, expected={EXPECTED[split]}')

        for seed in SEEDS:
            # Re-seed evaluation state identically before each EF model so clip choices are fixed.
            seed_everything(EVAL_SEED,deterministic=True)
            ef_dl=loader(ef_ds,args.batch_size_ef,args.num_workers,device)
            model=build_ef().to(device); ck=load_state(model,Path(args.ef_root)/f'seed_{seed}'/'best.pt',device)
            metrics=eval_ef(model,ef_dl,device); metrics.update({'seed':seed,'split':split,'model':'EF only','checkpoint_epoch':ck.get('epoch')})
            d=outroot/'ef_only'/f'seed_{seed}'; d.mkdir(parents=True,exist_ok=True)
            with (d/f'{split}_metrics.json').open('w') as f: json.dump(metrics,f,indent=2)
            del model; torch.cuda.empty_cache()

            seed_everything(EVAL_SEED,deterministic=True)
            seg_dl=loader(seg_ds,args.batch_size_seg,args.num_workers,device)
            model=build_seg().to(device); ck=load_state(model,Path(args.seg_root)/f'seed_{seed}'/'best.pt',device)
            metrics=eval_seg(model,seg_dl,device); metrics.update({'seed':seed,'split':split,'model':'Segmentation only','checkpoint_epoch':ck.get('epoch')})
            d=outroot/'segmentation_only'/f'seed_{seed}'; d.mkdir(parents=True,exist_ok=True)
            with (d/f'{split}_metrics.json').open('w') as f: json.dump(metrics,f,indent=2)
            del model; torch.cuda.empty_cache()

            seed_everything(EVAL_SEED,deterministic=True)
            mtl_dl=loader(mtl_ds,args.batch_size_mtl,args.num_workers,device)
            ckpath=Path(args.mtl_root)/f'seed_{seed}'/'best.pt'
            ck=torch.load(ckpath,map_location=device)
            cfg=ck.get('config',{})
            model=MultitaskDeepLabV3(pretrained=False,
                regression_hidden_dim=int(cfg.get('regression_hidden_dim',256)),
                dropout=float(cfg.get('dropout',0.3))).to(device)
            model.load_state_dict(ck['model_state_dict'])
            metrics=eval_mtl(model,mtl_dl,device); metrics.update({'seed':seed,'split':split,'model':'Multi-task','weight':'W2','checkpoint_epoch':ck.get('epoch')})
            d=outroot/'multitask_W2'/f'seed_{seed}'; d.mkdir(parents=True,exist_ok=True)
            with (d/f'{split}_metrics.json').open('w') as f: json.dump(metrics,f,indent=2)
            del model; torch.cuda.empty_cache()

        print(f'Completed final evaluation split: {split}')

    print(f'Final evaluation written to {outroot.resolve()}')


if __name__=='__main__': main()
