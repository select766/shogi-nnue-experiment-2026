#!/usr/bin/env python3
"""Convert search-utility details to hard winning-direction decision targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_cache(rows, minimum_improvement=10.0, score_scaling=361.0):
    output = np.empty((len(rows), 16), dtype=np.float32)
    changed = 0
    for index, row in enumerate(rows):
        utilities = np.asarray(row["utilities_cp"], dtype=np.float64)
        gates = np.asarray(row["gates"], dtype=np.float64)
        gain = float(utilities.max() - utilities[0])
        output[index, :8] = (utilities.max() - utilities[1:]) / score_scaling
        if gain >= minimum_improvement:
            winners = np.flatnonzero(utilities[1:] == utilities.max())
            target = np.zeros(8, dtype=np.float64)
            target[winners] = 1.0 / len(winners)
            changed += 1
        else:
            target = gates[0] / gates[0].sum()
        output[index, 8:] = target
    return output, changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--start-root", type=int, required=True)
    parser.add_argument("--num-roots", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_rows = [json.loads(line) for line in args.details.read_text().splitlines()]
    rows = all_rows[args.start_root : args.start_root + args.num_roots]
    if len(rows) != args.num_roots:
        parser.error("requested range exceeds details")
    cache, changed = build_cache(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as target:
        np.save(target, cache)
    temporary.replace(args.output)
    summary = {
        "format": "decision-aligned-teacher-v1",
        "roots": len(rows),
        "source_range": [args.start_root, args.start_root + len(rows)],
        "changed_roots": changed,
        "changed_fraction": changed / len(rows),
        "target": "uniform over utility-maximizing expert-axis directions when gain>=10cp; otherwise M0 gate",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
