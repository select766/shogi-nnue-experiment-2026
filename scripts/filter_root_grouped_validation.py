#!/usr/bin/env python3
"""Remove train-root overlap and duplicate roots from grouped validation data."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


RECORD_BYTES = 40


def root_sfens(path: Path):
    raw = path.read_bytes()
    if len(raw) % RECORD_BYTES:
        raise ValueError(f"invalid roots file: {path}")
    return [raw[index : index + 32] for index in range(0, len(raw), RECORD_BYTES)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train = Path(args.train).resolve()
    validation = Path(args.validation).resolve()
    output = Path(args.output).resolve()
    with (validation / "metadata.json").open() as file:
        metadata = json.load(file)
    group_size = int(metadata["group_size"])
    validation_sfens = root_sfens(validation / "roots.bin")
    train_set = set(root_sfens(train / "roots.bin"))
    seen = set()
    keep = []
    train_overlap = 0
    duplicates = 0
    for index, sfen in enumerate(validation_sfens):
        if sfen in train_set:
            train_overlap += 1
        elif sfen in seen:
            duplicates += 1
        else:
            keep.append(index)
            seen.add(sfen)
    keep = np.asarray(keep, dtype=np.int64)
    num_input = len(validation_sfens)
    roots = np.memmap(
        validation / "roots.bin",
        mode="r",
        dtype=np.uint8,
        shape=(num_input, RECORD_BYTES),
    )
    leaves = np.memmap(
        validation / "leaves.bin",
        mode="r",
        dtype=np.uint8,
        shape=(num_input, group_size, RECORD_BYTES),
    )
    offsets = np.load(validation / "offsets.npy", mmap_mode="r")
    output.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("roots.bin", roots[keep]),
        ("leaves.bin", leaves[keep]),
    ):
        temporary = output / f"{name}.tmp"
        with temporary.open("wb") as file:
            file.write(np.ascontiguousarray(values).tobytes())
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, output / name)
    np.save(output / "offsets.npy", np.asarray(offsets[keep]))
    filtered_metadata = dict(metadata)
    filtered_metadata["num_roots"] = int(keep.size)
    filtered_metadata["filter"] = {
        "source_validation_roots": num_input,
        "removed_train_sfen_overlap": train_overlap,
        "removed_validation_duplicate_sfen": duplicates,
    }
    with (output / "metadata.json").open("w") as file:
        json.dump(filtered_metadata, file, indent=2)
        file.write("\n")
    print(json.dumps(filtered_metadata["filter"], indent=2))
    print(f"retained_roots={keep.size}")


if __name__ == "__main__":
    main()
