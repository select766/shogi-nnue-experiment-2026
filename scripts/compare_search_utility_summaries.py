#!/usr/bin/env python3
"""Compare matched search-utility summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    identity_keys = [
        "roots", "start_root", "candidate_nodes", "reference_nodes",
        "logit_bias", "candidate_geometry", "minimum_improvement_cp",
    ]
    for key in identity_keys:
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"mismatched {key}: {baseline.get(key)} != {candidate.get(key)}")
    metrics = [
        "teacher_changed_fraction", "oracle_gain_cp_mean",
        "oracle_gain_cp_positive_fraction", "candidate_move_diversity_mean",
    ]
    deltas = {
        key: float(candidate[key] - baseline[key])
        for key in metrics
    }
    supported = (
        deltas["teacher_changed_fraction"] > 0
        and deltas["candidate_move_diversity_mean"] > 0
    )
    output = {
        "format": "search-utility-summary-comparison-v1",
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "matched_conditions": {key: baseline.get(key) for key in identity_keys},
        "baseline_metrics": {key: baseline[key] for key in metrics},
        "candidate_metrics": {key: candidate[key] for key in metrics},
        "candidate_minus_baseline": deltas,
        "classification": "supported" if supported else "rejected",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
