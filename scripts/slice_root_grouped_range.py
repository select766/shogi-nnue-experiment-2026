#!/usr/bin/env python3
"""Materialize a contiguous root range and its aligned teacher-cache rows."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


RECORD_BYTES = 40


def copy_range(source, target, offset, size):
    temporary = target.with_suffix(target.suffix + ".tmp")
    with source.open("rb") as input_file, temporary.open("wb") as output_file:
        input_file.seek(offset)
        remaining = size
        while remaining:
            block = input_file.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise EOFError(f"short source file: {source}")
            output_file.write(block)
            remaining -= len(block)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-root", type=int, required=True)
    parser.add_argument("--num-roots", type=int, required=True)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--output-teacher-cache", type=Path)
    args = parser.parse_args()
    if args.start_root < 0 or args.num_roots <= 0:
        parser.error("start-root must be non-negative and num-roots positive")
    if bool(args.teacher_cache) != bool(args.output_teacher_cache):
        parser.error("teacher-cache inputs and outputs must be specified together")
    metadata = json.loads((args.source / "metadata.json").read_text())
    source_roots = int(metadata["num_roots"])
    group_size = int(metadata["group_size"])
    stop = args.start_root + args.num_roots
    if stop > source_roots:
        parser.error("requested range exceeds source roots")
    args.output.mkdir(parents=True, exist_ok=True)
    copy_range(
        args.source / "roots.bin", args.output / "roots.bin",
        args.start_root * RECORD_BYTES, args.num_roots * RECORD_BYTES,
    )
    copy_range(
        args.source / "leaves.bin", args.output / "leaves.bin",
        args.start_root * group_size * RECORD_BYTES,
        args.num_roots * group_size * RECORD_BYTES,
    )
    output_metadata = dict(metadata)
    output_metadata["num_roots"] = args.num_roots
    output_metadata["slice"] = {
        "source": str(args.source.resolve()), "start_root": args.start_root,
        "stop_root_exclusive": stop,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n"
    )
    if args.teacher_cache:
        cache = np.load(args.teacher_cache, mmap_mode="r")
        if cache.ndim != 2 or cache.shape[0] < stop:
            raise ValueError("teacher cache does not cover requested range")
        args.output_teacher_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_teacher_cache.with_suffix(
            args.output_teacher_cache.suffix + ".tmp"
        )
        with temporary.open("wb") as target:
            np.save(target, np.asarray(cache[args.start_root:stop]))
        os.replace(temporary, args.output_teacher_cache)
    print(json.dumps(output_metadata["slice"], indent=2))


if __name__ == "__main__":
    main()
