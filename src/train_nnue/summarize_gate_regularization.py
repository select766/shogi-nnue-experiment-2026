"""Summarize TensorBoard scalars from the gate regularization sweep."""

import argparse
import json
import os
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = (
    "val_loss",
    "val_regularized_loss",
    "val_gate_entropy",
    "val_effective_experts",
    "val_gate_max_weight",
    "val_gate_top2_mass",
    "val_balance_kl",
    "val_router_loss",
    "val_router_top1_match",
    "val_router_teacher_top1_match",
    "val_router_expected_match",
    "val_router_regret",
    "val_router_teacher_entropy",
)


def load_run(run_directory):
    version_directory = run_directory / "lightning_logs" / "version_0"
    event_files = sorted(version_directory.glob("events.out.tfevents.*"))
    if not event_files:
        return None

    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    available_tags = set(accumulator.Tags().get("scalars", []))
    histories = {}
    for tag in TAGS:
        if tag in available_tags:
            histories[tag] = [
                {"step": value.step, "value": value.value}
                for value in accumulator.Scalars(tag)
            ]

    hyperparameters_path = version_directory / "hparams.yaml"
    hyperparameters = {}
    if hyperparameters_path.is_file():
        with hyperparameters_path.open() as file:
            hyperparameters = yaml.safe_load(file) or {}

    val_losses = histories.get("val_loss", [])
    best_index = None
    if val_losses:
        best_index = min(range(len(val_losses)), key=lambda index: val_losses[index]["value"])

    def values_at(index):
        if index is None:
            return None
        return {
            tag: values[index]["value"]
            for tag, values in histories.items()
            if index < len(values)
        }

    return {
        "run_name": run_directory.name,
        "lambda_sparse": hyperparameters.get("lambda_sparse"),
        "lambda_balance": hyperparameters.get("lambda_balance"),
        "completed_validation_epochs": len(val_losses),
        "best_epoch_index": best_index,
        "best": values_at(best_index),
        "final": values_at(len(val_losses) - 1) if val_losses else None,
        "histories": histories,
        "event_file": str(event_files[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--pattern", default="gate_reg_*_from510")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logs_root = Path(args.logs_root).resolve()
    runs = []
    for run_directory in sorted(logs_root.glob(args.pattern)):
        summary = load_run(run_directory)
        if summary is not None:
            runs.append(summary)

    output = {
        "logs_root": str(logs_root),
        "pattern": args.pattern,
        "runs": runs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(f"{output_path}.tmp")
    with temporary_path.open("w") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    os.replace(temporary_path, output_path)
    print(f"Wrote {len(runs)} runs to {output_path}")


if __name__ == "__main__":
    main()
