#!/usr/bin/env python3
"""Verify the professor-required matched B1/B3 settings before test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from echonet.utils.stage1_corrected import SEEDS

MATCH_FIELDS = [
    ("ef_input", "T"),
    ("ef_input", "period"),
    ("ef_input", "training_sampling"),
    ("ef_input", "validation_sampling"),
    ("split", None),
    ("preprocessing", None),
    ("ef_architecture", "encoder"),
    ("ef_architecture", "temporal_module"),
    ("ef_architecture", "aggregation"),
    ("training", "epochs"),
    ("training", "batch_size"),
    ("training", "optimizer"),
    ("training", "learning_rate"),
    ("training", "momentum"),
    ("training", "weight_decay"),
    ("training", "lr_step_period"),
    ("checkpoint_rule", None),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", default="output/stage1/corrected")
    p.add_argument("--results-root", default="results/stage1/corrected")
    return p.parse_args()


def load(path):
    with path.open() as f:
        return json.load(f)


def get(cfg, group, key):
    value = cfg.get(group) if group in cfg else None
    if key is None:
        return value
    if not isinstance(value, dict):
        return None
    return value.get(key)


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    results_root = Path(args.results_root)
    lock_path = results_root / "validation_weight_selection" / "LOCKED_WEIGHT.txt"
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)
    locked = lock_path.read_text().strip()

    comparisons = []
    for seed in SEEDS:
        b1_path = run_root / "B1_video_ef" / f"seed_{seed}" / "run_config.json"
        b3_path = run_root / "B3_video_mtl" / locked / f"seed_{seed}" / "run_config.json"
        b1, b3 = load(b1_path), load(b3_path)
        for group, key in MATCH_FIELDS:
            a, b = get(b1, group, key), get(b3, group, key)
            field = group if key is None else f"{group}.{key}"
            comparisons.append({"seed": seed, "field": field, "B1": a, "B3": b, "match": a == b})
            if a != b:
                raise RuntimeError(f"B1/B3 mismatch for seed {seed}, {field}: {a!r} != {b!r}")

        if b1.get("dataset", {}).get("volume_tracings_required") is not False:
            raise RuntimeError(f"B1 seed {seed} is not marked video-only")
        if b3.get("oracle_guard") is None:
            raise RuntimeError(f"B3 seed {seed} lacks oracle guard evidence")

    out = results_root / "verification" / "matched_B1_B3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pass",
        "locked_weight": locked,
        "seeds": list(SEEDS),
        "all_required_B1_B3_fields_match": True,
        "comparisons": comparisons,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
