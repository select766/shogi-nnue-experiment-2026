#!/usr/bin/env python3
"""Export Lightning epoch metrics and a provisional saturation decision."""

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalar_events(accumulator, tag):
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [
        {"step": int(event.step), "value": float(event.value)}
        for event in accumulator.Scalars(tag)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-improvement", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if args.min_improvement < 0:
        raise ValueError("min improvement must be non-negative")
    if args.patience <= 0:
        raise ValueError("patience must be positive")

    log_dir = Path(args.log_dir).resolve()
    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()
    metrics = {
        tag: scalar_events(accumulator, tag)
        for tag in ("val_loss", "train_loss_epoch", "train_loss", "lr")
    }
    validation = metrics["val_loss"]
    best_index = None
    best_value = None
    meaningful_best_index = None
    meaningful_best_value = None
    for index, event in enumerate(validation):
        if best_value is None or event["value"] < best_value:
            best_index = index
            best_value = event["value"]
        if (
            meaningful_best_value is None
            or event["value"] <= meaningful_best_value - args.min_improvement
        ):
            meaningful_best_index = index
            meaningful_best_value = event["value"]

    epochs_after_best = (
        len(validation) - meaningful_best_index - 1
        if meaningful_best_index is not None
        else 0
    )
    saturated = (
        meaningful_best_index is not None and epochs_after_best >= args.patience
    )
    output = {
        "log_dir": str(log_dir),
        "completed_validation_epochs": len(validation),
        "metrics": metrics,
        "provisional_saturation_rule": {
            "min_improvement": args.min_improvement,
            "patience": args.patience,
            "definition": (
                "No new validation minimum by at least min_improvement for "
                "patience completed epochs after the last meaningful "
                "improvement. Final model selection additionally requires "
                "paired root-level comparison."
            ),
            "best_validation_index": best_index,
            "best_validation_loss": best_value,
            "last_meaningful_improvement_index": meaningful_best_index,
            "last_meaningful_improvement_loss": meaningful_best_value,
            "epochs_after_best": epochs_after_best,
            "saturated": saturated,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
