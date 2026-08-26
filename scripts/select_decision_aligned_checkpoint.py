#!/usr/bin/env python3
"""Apply the pre-registered CE/group-loss screening to training curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def values_by_index(curve, metric):
    return [event["value"] for event in curve["metrics"].get(metric, [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-curve", type=Path, required=True)
    parser.add_argument("--candidate", nargs=3, action="append", metavar=("NAME", "CURVE", "CHECKPOINT_DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = json.loads(args.control_curve.read_text())
    control_losses = values_by_index(control, "val_mean_group_loss")
    if not control_losses:
        raise ValueError("control curve has no val_mean_group_loss")
    loss_limit = min(control_losses) + 0.0001
    candidates = []
    for name, curve_path, checkpoint_dir in args.candidate:
        curve = json.loads(Path(curve_path).read_text())
        losses = values_by_index(curve, "val_mean_group_loss")
        cross_entropies = values_by_index(curve, "val_router_loss")
        if len(losses) != len(cross_entropies):
            raise ValueError(f"unaligned curve metrics for {name}")
        eligible = [
            index for index, loss in enumerate(losses) if loss <= loss_limit
        ]
        selected = min(eligible, key=lambda index: (cross_entropies[index], index)) if eligible else None
        candidates.append(
            {
                "name": name,
                "selected_epoch": selected,
                "checkpoint": str(Path(checkpoint_dir) / f"{selected}.ckpt") if selected is not None else None,
                "val_mean_group_loss": losses[selected] if selected is not None else None,
                "val_router_loss": cross_entropies[selected] if selected is not None else None,
            }
        )
    eligible_candidates = [item for item in candidates if item["selected_epoch"] is not None]
    selected = min(eligible_candidates, key=lambda item: (item["val_router_loss"], item["name"])) if eligible_candidates else None
    output = {
        "format": "decision-aligned-selection-v1",
        "control_loss_min": min(control_losses),
        "candidate_loss_limit": loss_limit,
        "candidates": candidates,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
