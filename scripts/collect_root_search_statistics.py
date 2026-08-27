#!/usr/bin/env python3
"""Collect aligned static plus shallow-search feature caches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time

import cshogi
import numpy as np

try:
    from scripts.build_root_feature_cache import phase_features
    from scripts.collect_search_leaf_groups import EngineProcess, RECORD_BYTES, record_to_sfen
except ModuleNotFoundError:
    from build_root_feature_cache import phase_features
    from collect_search_leaf_groups import EngineProcess, RECORD_BYTES, record_to_sfen
from train_nnue.root_search_statistics import FEATURE_NAMES, shallow_multipv_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots-file", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--max-roots", type=int, required=True)
    parser.add_argument("--nodes", type=int, default=1024)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    available = args.roots_file.stat().st_size // RECORD_BYTES
    if (
        args.roots_file.stat().st_size % RECORD_BYTES
        or not 0 < args.max_roots <= available
        or args.nodes <= 0
        or not 1 <= args.multipv <= 4
        or args.workers <= 0
    ):
        parser.error("invalid root/search/worker parameters")
    records = []
    with args.roots_file.open("rb") as source:
        for index in range(args.max_roots):
            records.append((index, source.read(RECORD_BYTES)))
    progress = 0
    lock = threading.Lock()
    started = time.monotonic()

    def process(chunk):
        nonlocal progress
        engine = EngineProcess([str(args.engine.resolve())])
        engine.initialize(
            [
                ("Threads", 1),
                ("USI_Hash", args.hash_mb),
                ("EvalDir", args.eval_dir.resolve()),
                ("USI_OwnBook", "false"),
                ("MultiPV", args.multipv),
            ]
        )
        board = cshogi.Board()
        psfen = np.zeros(1, dtype=cshogi.PackedSfen)
        output = []
        try:
            for index, record in chunk:
                sfen = record_to_sfen(record, board, psfen)
                board.set_sfen(sfen)
                static = phase_features(board, board.move_number)
                search = shallow_multipv_features(
                    engine, sfen, nodes=args.nodes, multipv=args.multipv
                )
                output.append((index, np.concatenate([static, search])))
                with lock:
                    progress += 1
                    if progress % 1000 == 0 or progress == args.max_roots:
                        elapsed = time.monotonic() - started
                        print(
                            json.dumps(
                                {
                                    "completed": progress,
                                    "total": args.max_roots,
                                    "roots_per_second": progress / elapsed,
                                }
                            ),
                            flush=True,
                        )
        finally:
            engine.close()
        return output

    chunks = [[] for _ in range(args.workers)]
    for item in records:
        chunks[item[0] % args.workers].append(item)
    collected = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, chunk) for chunk in chunks if chunk]
        for future in as_completed(futures):
            collected.extend(future.result())
    collected.sort(key=lambda item: item[0])
    output = np.stack([item[1] for item in collected]).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as target:
        np.save(target, output)
    os.replace(temporary, args.output)
    metadata = {
        "format": "root-search-statistics-v1",
        "features": [
            "game_ply", "board_piece_fraction", "hand_piece_fraction",
            "promoted_fraction", "legal_move_fraction", "material_count_balance",
            "king_distance", *FEATURE_NAMES,
        ],
        "static_dimension": 7,
        "search_dimension": len(FEATURE_NAMES),
        "dimension": int(output.shape[1]),
        "roots": args.max_roots,
        "nodes": args.nodes,
        "multipv": args.multipv,
        "source": str(args.roots_file.resolve()),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
