#!/usr/bin/env python3
# HISTORICAL T0 CONTROLLED-BASELINE CHECK. Use verify_stage1_corrected_runs.py.
"""Verify identical relevant initialization versus MultitaskDeepLabV3."""
import json, torch
from echonet.modeling.multitask_deeplab import MultitaskDeepLabV3
from echonet.modeling.stage1_controlled_baselines import EFOnlyDeepLabV3, SegmentationOnlyDeepLabV3
from echonet.utils.reproducibility import seed_everything

def check(label,a,b):
    assert set(a)==set(b), f"{label} keys differ"
    maxd=0.0
    for k in a:
        x=a[k].detach().cpu(); y=b[k].detach().cpu()
        if torch.is_floating_point(x):
            d=float((x-y).abs().max()) if x.numel() else 0.0
            assert d==0.0, f"{label}:{k} differs by {d}"
            maxd=max(maxd,d)
        else:
            assert torch.equal(x,y), f"{label}:{k} differs"
    return maxd

seed=42; kwargs=dict(pretrained=True,regression_hidden_dim=256,dropout=0.3)
seed_everything(seed,True); m=MultitaskDeepLabV3(**kwargs)
back={k:v.clone() for k,v in m.base_model.backbone.state_dict().items()}
cls={k:v.clone() for k,v in m.base_model.classifier.state_dict().items()}
efh={k:v.clone() for k,v in m.ef_head.state_dict().items()}
del m
seed_everything(seed,True); e=EFOnlyDeepLabV3(**kwargs)
d1=check("EF backbone",back,e.backbone.state_dict()); d2=check("EF head",efh,e.ef_head.state_dict()); del e
seed_everything(seed,True); s=SegmentationOnlyDeepLabV3(**kwargs)
d3=check("Seg backbone",back,s.backbone.state_dict()); d4=check("Seg classifier",cls,s.classifier.state_dict()); del s
print(json.dumps({"seed":seed,"ef_backbone_diff":d1,"ef_head_diff":d2,"seg_backbone_diff":d3,"seg_classifier_diff":d4,"PASS":True},indent=2))
