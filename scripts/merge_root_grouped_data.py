#!/usr/bin/env python3
"""Concatenate disjoint root-grouped datasets after validating their format."""

import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    metadata = [json.loads((path / "metadata.json").read_text()) for path in args.input_dir]
    reference = metadata[0]
    keys = (
        "format", "group_size", "record_bytes", "leaf_distribution",
        "search_threads", "search_nodes", "search_seed", "search_engine",
        "search_eval_dir", "label_engine", "label_eval_dir", "label",
    )
    for path, item in zip(args.input_dir, metadata):
        if item.get("status") != "complete":
            raise ValueError(f"incomplete input: {path}")
        for key in keys:
            if item.get(key) != reference.get(key):
                raise ValueError(f"incompatible {key}: {path}")

    args.output_dir.mkdir(parents=True)
    record_bytes = int(reference["record_bytes"])
    group_size = int(reference["group_size"])
    total_roots = 0
    for name, multiplier in (("roots.bin", 1), ("leaves.bin", group_size)):
        with (args.output_dir / name).open("wb") as output:
            for path, item in zip(args.input_dir, metadata):
                expected = int(item["num_roots"]) * multiplier * record_bytes
                source_path = path / name
                if source_path.stat().st_size != expected:
                    raise ValueError(f"size mismatch: {source_path}")
                with source_path.open("rb") as source:
                    shutil.copyfileobj(source, output)
    total_roots = sum(int(item["num_roots"]) for item in metadata)
    merged = {
        **{key: reference.get(key) for key in keys},
        "num_roots": total_roots,
        "source_chunks": [str(path.resolve()) for path in args.input_dir],
        "source_start_roots": [item.get("source_start_root") for item in metadata],
        "source_records_consumed": sum(
            int(item.get("source_records_consumed", 0)) for item in metadata
        ),
        "skipped_roots": sum(int(item.get("skipped_roots", 0)) for item in metadata),
        "mean_qsearch_boundary_visits": sum(
            float(item["mean_qsearch_boundary_visits"]) * int(item["num_roots"])
            for item in metadata
        ) / total_roots,
        "elapsed_seconds_sum": sum(float(item.get("elapsed_seconds", 0)) for item in metadata),
        "status": "complete",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
