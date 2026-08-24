#!/usr/bin/env python3
"""Relabel grouped quiet leaves with an independent fixed-node search teacher."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time

import cshogi
import numpy as np

try:
    from scripts.collect_search_leaf_groups import (
        EngineProcess,
        RECORD_BYTES,
        endpoint_record,
        record_to_sfen,
    )
except ModuleNotFoundError:
    from collect_search_leaf_groups import (
        EngineProcess,
        RECORD_BYTES,
        endpoint_record,
        record_to_sfen,
    )


def parse_score_and_pv(line):
    """Return a root-side score and PV from one standard USI info line."""
    fields = line.split()
    if not fields or fields[0] != "info" or "score" not in fields:
        return None
    index = fields.index("score")
    if index + 2 >= len(fields):
        return None
    kind = fields[index + 1]
    value_token = fields[index + 2]
    try:
        value = int(value_token)
    except ValueError:
        return None
    if kind == "mate":
        # YaneuraOu uses `mate -0` for a position where the side to move is
        # already mated.  int("-0") loses that semantically important sign.
        sign = -1 if value_token.startswith("-") else 1
        score = (32000 - min(abs(value), 1000)) * sign
    elif kind == "cp":
        score = max(-32000, min(32000, value))
    else:
        return None
    pv = fields[fields.index("pv") + 1 :] if "pv" in fields else []
    return score, pv


def fixed_node_search(labeler, sfen, nodes):
    labeler.send(f"position sfen {sfen}")
    labeler.send(f"go nodes {nodes}")
    latest = None
    bestmove = None
    while True:
        line = labeler.read_until(lambda _: True)
        parsed = parse_score_and_pv(line)
        if parsed is not None:
            latest = parsed
        if line.startswith("bestmove "):
            bestmove = line.split()[1]
            break
    if latest is None:
        raise ValueError(f"search returned no score for {sfen}")
    score, pv = latest
    if not pv and bestmove not in {None, "resign", "win", "none"}:
        pv = [bestmove]
    return endpoint_record(sfen, score, pv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-engine", type=Path, required=True)
    parser.add_argument("--label-eval-dir", type=Path, required=True)
    parser.add_argument("--label-nodes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    if args.label_nodes <= 0 or args.workers <= 0:
        parser.error("label-nodes and workers must be positive")
    metadata = json.loads((args.input / "metadata.json").read_text())
    num_roots = int(metadata["num_roots"])
    group_size = int(metadata["group_size"])
    total = num_roots * group_size
    expected_size = total * RECORD_BYTES
    if (args.input / "leaves.bin").stat().st_size != expected_size:
        raise ValueError("input leaves.bin does not match metadata")
    args.output.mkdir(parents=True, exist_ok=True)
    roots_target = args.output / "roots.bin"
    if not roots_target.exists():
        os.link(args.input / "roots.bin", roots_target)
    leaves_tmp = args.output / "leaves.bin.tmp"
    if leaves_tmp.exists() or (args.output / "leaves.bin").exists():
        raise FileExistsError("output leaves already exist")

    local = threading.local()
    engines = []
    engine_lock = threading.Lock()

    def initialize_worker():
        engine = EngineProcess([str(args.label_engine.resolve())])
        engine.initialize(
            [
                ("Threads", 1),
                ("USI_Hash", args.hash_mb),
                ("EvalDir", args.label_eval_dir.resolve()),
                ("USI_OwnBook", "false"),
            ]
        )
        local.engine = engine
        local.board = cshogi.Board()
        local.psfen = np.zeros(1, dtype=cshogi.PackedSfen)
        with engine_lock:
            engines.append(engine)

    def relabel(record):
        sfen = record_to_sfen(record, local.board, local.psfen)
        return fixed_node_search(local.engine, sfen, args.label_nodes)

    def records(source):
        for _ in range(total):
            record = source.read(RECORD_BYTES)
            if len(record) != RECORD_BYTES:
                raise EOFError("short input leaves.bin")
            yield record

    started = time.monotonic()
    try:
        with (args.input / "leaves.bin").open("rb") as source, \
                leaves_tmp.open("wb") as target, ThreadPoolExecutor(
                    max_workers=args.workers, initializer=initialize_worker
                ) as executor:
            for index, output_record in enumerate(
                executor.map(relabel, records(source)), start=1
            ):
                target.write(output_record)
                if index % args.progress_every == 0:
                    elapsed = time.monotonic() - started
                    rate = index / elapsed
                    print(json.dumps({
                        "completed_leaves": index,
                        "leaves_per_second": rate,
                        "eta_seconds": (total - index) / rate,
                    }), flush=True)
            target.flush()
            os.fsync(target.fileno())
    finally:
        for engine in engines:
            engine.close()
    os.replace(leaves_tmp, args.output / "leaves.bin")
    output_metadata = dict(metadata)
    output_metadata["source_label"] = metadata.get("label")
    output_metadata["label"] = (
        "fixed-node search score at input quiet leaf, sign-adjusted at teacher "
        "PV endpoint"
    )
    output_metadata["label_engine"] = str(args.label_engine.resolve())
    output_metadata["label_eval_dir"] = str(args.label_eval_dir.resolve())
    output_metadata["search_relabel"] = {
        "label_engine": str(args.label_engine.resolve()),
        "label_eval_dir": str(args.label_eval_dir.resolve()),
        "label_nodes": args.label_nodes,
        "workers": args.workers,
        "label": output_metadata["label"],
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n"
    )
    print(json.dumps(output_metadata["search_relabel"], indent=2))


if __name__ == "__main__":
    main()
