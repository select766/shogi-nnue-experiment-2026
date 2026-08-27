#!/usr/bin/env python3
"""Assign one shared restricted-search utility to every pooled candidate move."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

try:
    from scripts.collect_search_leaf_groups import EngineProcess
    from scripts.collect_search_utility_teachers import reference_utilities
except ModuleNotFoundError:
    from collect_search_leaf_groups import EngineProcess
    from collect_search_utility_teachers import reference_utilities


def load_aligned(paths):
    groups = [
        [json.loads(line) for line in Path(path).read_text().splitlines()]
        for path in paths
    ]
    roots = [row["source_root"] for row in groups[0]]
    for rows in groups[1:]:
        if [row["source_root"] for row in rows] != roots:
            raise ValueError("detail files do not contain the same ordered roots")
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", action="append", required=True)
    parser.add_argument("--reference-engine", type=Path, required=True)
    parser.add_argument("--reference-eval-dir", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.nodes, args.workers, args.hash_mb) <= 0:
        parser.error("node/worker/hash values must be positive")
    groups = load_aligned(args.details)
    jobs = []
    for index, aligned in enumerate(zip(*groups)):
        sfen = aligned[0]["sfen"]
        if any(row["sfen"] != sfen for row in aligned):
            raise ValueError("SFEN differs across aligned detail rows")
        moves = list(dict.fromkeys(move for row in aligned for move in row["moves"]))
        jobs.append((index, int(aligned[0]["source_root"]), sfen, moves))

    def process(chunk):
        engine = EngineProcess([str(args.reference_engine.resolve())])
        engine.initialize([
            ("Threads", 1), ("USI_Hash", args.hash_mb),
            ("EvalDir", args.reference_eval_dir.resolve()), ("USI_OwnBook", "false"),
        ])
        rows = []
        try:
            for index, source_root, sfen, moves in chunk:
                _, scores = reference_utilities(engine, sfen, moves, args.nodes)
                rows.append((index, {
                    "source_root": source_root,
                    "sfen": sfen,
                    "unique_move_scores_cp": scores,
                }))
        finally:
            engine.close()
        return rows

    chunks = [[] for _ in range(args.workers)]
    for job in jobs:
        chunks[job[0] % args.workers].append(job)
    output = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed(executor.submit(process, chunk) for chunk in chunks if chunk):
            output.extend(future.result())
    output.sort(key=lambda item: item[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row) + "\n" for _, row in output)
    )


if __name__ == "__main__":
    main()
