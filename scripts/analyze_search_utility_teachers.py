#!/usr/bin/env python3
"""Summarize search-utility details for an exact contiguous root range."""

import argparse
import json
from pathlib import Path

import numpy as np


def quantiles(values):
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--start-root", type=int, default=0)
    parser.add_argument("--num-roots", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.details.read_text().splitlines()]
    stop = args.start_root + args.num_roots
    rows = rows[args.start_root:stop]
    if len(rows) != args.num_roots:
        parser.error("requested range exceeds details rows")
    gains = np.asarray([row["oracle_gain_cp"] for row in rows], dtype=np.float64)
    changed = np.asarray([row["teacher_changed"] for row in rows])
    diversities = np.asarray([len(set(row["moves"])) for row in rows])
    selected = np.asarray([row["selected_candidate"] for row in rows])
    cross_entropies = []
    l1_changes = []
    for row in rows:
        gates = np.asarray(row["gates"], dtype=np.float64)
        if row["teacher_changed"]:
            best = max(row["utilities_cp"])
            winners = np.flatnonzero(np.asarray(row["utilities_cp"]) == best)
            teacher = gates[winners].mean(axis=0)
        else:
            teacher = gates[0]
        teacher /= teacher.sum()
        base = gates[0] / gates[0].sum()
        cross_entropies.append(float(-(teacher * np.log(base + 1e-12)).sum()))
        l1_changes.append(float(np.abs(teacher - base).sum()))
    output = {
        "roots": args.num_roots,
        "range": [args.start_root, stop],
        "teacher_changed_fraction": float(changed.mean()),
        "oracle_gain_cp": {
            "mean": float(gains.mean()), "quantiles": quantiles(gains),
            "at_least_10_fraction": float((gains >= 10.0).mean()),
        },
        "candidate_move_diversity": {
            "mean": float(diversities.mean()), "quantiles": quantiles(diversities),
        },
        "selected_candidate_counts": {
            str(index): int((selected == index).sum()) for index in range(9)
        },
        "base_to_teacher_cross_entropy_mean": float(np.mean(cross_entropies)),
        "base_to_teacher_l1_change_mean": float(np.mean(l1_changes)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
