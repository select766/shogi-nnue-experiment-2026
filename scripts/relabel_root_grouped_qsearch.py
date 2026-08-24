#!/usr/bin/env python3
"""Relabel every grouped leaf with one fixed engine's exact qsearch value."""

import argparse
import json
import os
from pathlib import Path
import time

import cshogi
import numpy as np

try:
    from scripts.collect_search_leaf_groups import (
        EngineProcess,
        RECORD_BYTES,
        exact_qsearch,
        record_to_sfen,
    )
except ModuleNotFoundError:
    from collect_search_leaf_groups import (
        EngineProcess,
        RECORD_BYTES,
        exact_qsearch,
        record_to_sfen,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-engine", type=Path, required=True)
    parser.add_argument("--label-eval-dir", type=Path, required=True)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    metadata = json.loads((args.input / "metadata.json").read_text())
    num_roots = int(metadata["num_roots"])
    group_size = int(metadata["group_size"])
    expected_size = num_roots * group_size * RECORD_BYTES
    if (args.input / "leaves.bin").stat().st_size != expected_size:
        raise ValueError("input leaves.bin does not match metadata")
    args.output.mkdir(parents=True, exist_ok=True)
    roots_target = args.output / "roots.bin"
    if not roots_target.exists():
        os.link(args.input / "roots.bin", roots_target)
    leaves_tmp = args.output / "leaves.bin.tmp"
    if leaves_tmp.exists() or (args.output / "leaves.bin").exists():
        raise FileExistsError("output leaves already exist")

    labeler = EngineProcess([str(args.label_engine.resolve())])
    labeler.initialize(
        [
            ("Threads", 1),
            ("USI_Hash", args.hash_mb),
            ("EvalDir", args.label_eval_dir.resolve()),
            ("USI_OwnBook", "false"),
        ]
    )
    board = cshogi.Board()
    psfen = np.zeros(1, dtype=cshogi.PackedSfen)
    total = num_roots * group_size
    started = time.monotonic()
    try:
        with (args.input / "leaves.bin").open("rb") as source, leaves_tmp.open("wb") as target:
            for index in range(total):
                record = source.read(RECORD_BYTES)
                if len(record) != RECORD_BYTES:
                    raise EOFError("short input leaves.bin")
                sfen = record_to_sfen(record, board, psfen)
                target.write(exact_qsearch(labeler, sfen))
                if (index + 1) % args.progress_every == 0:
                    elapsed = time.monotonic() - started
                    rate = (index + 1) / elapsed
                    print(
                        json.dumps(
                            {
                                "completed_leaves": index + 1,
                                "leaves_per_second": rate,
                                "eta_seconds": (total - index - 1) / rate,
                            }
                        ),
                        flush=True,
                    )
            target.flush()
            os.fsync(target.fileno())
    finally:
        labeler.close()
    os.replace(leaves_tmp, args.output / "leaves.bin")
    output_metadata = dict(metadata)
    output_metadata["qsearch_relabel"] = {
        "label_engine": str(args.label_engine.resolve()),
        "label_eval_dir": str(args.label_eval_dir.resolve()),
        "label": "exact qsearch score at input leaf, sign-adjusted at qsearch PV endpoint",
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n"
    )
    print(json.dumps(output_metadata["qsearch_relabel"], indent=2))


if __name__ == "__main__":
    main()
