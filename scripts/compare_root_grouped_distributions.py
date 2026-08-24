#!/usr/bin/env python3
"""Compare two grouped leaf datasets aligned to identical root records."""

import argparse
import json
from pathlib import Path

import numpy as np


RECORD_BYTES = 40
POSITION_BYTES = 32


def load(path):
    metadata = json.loads((path / "metadata.json").read_text())
    roots = np.fromfile(path / "roots.bin", dtype=np.dtype(("V", RECORD_BYTES)))
    leaves = np.fromfile(path / "leaves.bin", dtype=np.uint8).reshape(
        len(roots), int(metadata["group_size"]), RECORD_BYTES
    )
    return metadata, roots, leaves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first_meta, first_roots, first = load(args.first)
    second_meta, second_roots, second = load(args.second)
    if len(first_roots) != len(second_roots) or not np.array_equal(
        first_roots, second_roots
    ):
        raise ValueError("datasets are not aligned to identical roots")

    same_root_any = []
    jaccards = []
    matched_fraction = []
    for left, right in zip(first[:, :, :POSITION_BYTES], second[:, :, :POSITION_BYTES]):
        left_set = {bytes(value) for value in left}
        right_set = {bytes(value) for value in right}
        intersection = len(left_set & right_set)
        same_root_any.append(intersection > 0)
        jaccards.append(intersection / len(left_set | right_set))
        matched_fraction.append(intersection / len(left_set))

    def score_summary(records):
        scores = records[:, :, 32:34].copy().view("<i2").reshape(-1).astype(float)
        return {
            "mean": float(scores.mean()),
            "mean_absolute": float(np.abs(scores).mean()),
            "quantiles": {
                str(q): float(np.quantile(scores, q))
                for q in (0.0, 0.1, 0.5, 0.9, 1.0)
            },
        }

    output = {
        "roots": len(first_roots),
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "first_policy": first_meta.get("search_expert_blending_dir"),
        "second_policy": second_meta.get("search_expert_blending_dir"),
        "same_root_any_leaf_overlap": float(np.mean(same_root_any)),
        "mean_same_root_jaccard": float(np.mean(jaccards)),
        "mean_first_unique_leaves_matched": float(np.mean(matched_fraction)),
        "first_global_unique_fraction": len({
            bytes(value) for value in first[:, :, :POSITION_BYTES].reshape(-1, POSITION_BYTES)
        }) / (first.shape[0] * first.shape[1]),
        "second_global_unique_fraction": len({
            bytes(value) for value in second[:, :, :POSITION_BYTES].reshape(-1, POSITION_BYTES)
        }) / (second.shape[0] * second.shape[1]),
        "first_score": score_summary(first),
        "second_score": score_summary(second),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
