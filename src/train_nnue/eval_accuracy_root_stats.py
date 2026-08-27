"""Evaluate an Expert Blending model with online shallow-root statistics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import threading
import time

import cshogi
from cshogi.usi import Engine
import numpy as np

from scripts.build_root_feature_cache import phase_features
from scripts.collect_search_leaf_groups import EngineProcess
from train_nnue.accuracy_statistics import accuracy_summary, stratified_summary
from train_nnue.eval_accuracy import load_dataset, resolve_engine_option_paths, resolve_path
from train_nnue.root_search_statistics import shallow_multipv_features


def create_candidate(path, options):
    engine = Engine(path)
    for key, value in options.items():
        engine.setoption(key, value)
    engine.isready()
    return engine


def create_shallow(path, options, multipv):
    engine = EngineProcess([path])
    resolved = [(key, value) for key, value in options.items()]
    resolved.append(("MultiPV", multipv))
    engine.initialize(resolved)
    return engine


def worker_fn(positions, config, project_root, progress_callback):
    candidate_path = resolve_path(config["engine_path"], project_root)
    candidate_options = resolve_engine_option_paths(
        config.get("engine_options", {}), project_root
    )
    shallow_path = resolve_path(config["shallow_engine_path"], project_root)
    shallow_options = resolve_engine_option_paths(
        config.get("shallow_engine_options", {}), project_root
    )
    nodes = int(config.get("shallow_nodes", 1024))
    multipv = int(config.get("shallow_multipv", 4))
    candidate = create_candidate(candidate_path, candidate_options)
    shallow = create_shallow(shallow_path, shallow_options, multipv)
    board = cshogi.Board()
    results = []
    shallow_seconds = 0.0
    main_seconds = 0.0
    try:
        for index, record in positions:
            sfen = record["sfen"]
            started = time.monotonic()
            search_features = shallow_multipv_features(
                shallow, sfen, nodes=nodes, multipv=multipv
            )
            shallow_seconds += time.monotonic() - started
            board.set_sfen(sfen)
            static = phase_features(board, board.move_number)
            auxiliary = np.concatenate([static, search_features])
            encoded = ",".join(f"{float(value):.8g}" for value in auxiliary[7:])
            candidate.setoption("ExpertBlendingRootSearchStatistics", encoded)
            candidate.position(sfen=f"sfen {sfen}")
            started = time.monotonic()
            bestmove, _ = candidate.go(**config.get("go_params", {"nodes": 1000000}))
            main_seconds += time.monotonic() - started
            results.append(
                {
                    "index": index,
                    "sfen": sfen,
                    "expected": record["bestmove"],
                    "actual": bestmove,
                    "match": bestmove == record["bestmove"],
                }
            )
            progress_callback()
    finally:
        shallow.close()
        candidate.quit()
    return results, shallow_seconds, main_seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--go-nodes", type=int)
    args = parser.parse_args()
    project_root = os.path.abspath(args.project_root)
    config = json.loads(Path(args.config).read_text())
    if args.go_nodes is not None:
        if args.go_nodes <= 0:
            parser.error("--go-nodes must be positive")
        config["go_params"] = {"nodes": args.go_nodes}
    records = load_dataset(args.dataset)
    if args.max_positions is not None:
        if args.max_positions <= 0:
            parser.error("--max-positions must be positive")
        records = records[: args.max_positions]
    workers = int(config.get("num_workers", 4))
    chunks = [[] for _ in range(workers)]
    for item in enumerate(records):
        chunks[item[0] % workers].append(item)
    lock = threading.Lock()
    progress = 0

    def update_progress():
        nonlocal progress
        with lock:
            progress += 1
            if progress % 100 == 0 or progress == len(records):
                print(f"Progress: {progress}/{len(records)}", file=sys.stderr)

    all_results = []
    shallow_seconds = 0.0
    main_seconds = 0.0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(worker_fn, chunk, config, project_root, update_progress)
            for chunk in chunks
            if chunk
        ]
        for future in as_completed(futures):
            results, shallow_time, main_time = future.result()
            all_results.extend(results)
            shallow_seconds += shallow_time
            main_seconds += main_time
    all_results.sort(key=lambda row: row["index"])
    matches = [row["match"] for row in all_results]
    summary = accuracy_summary(matches)
    output = {
        **summary,
        "strata": stratified_summary(records, matches),
        "timing": {
            "shallow_seconds_sum": shallow_seconds,
            "main_seconds_sum": main_seconds,
            "mean_shallow_ms": 1000.0 * shallow_seconds / len(records),
            "mean_main_ms": 1000.0 * main_seconds / len(records),
            "added_time_fraction": shallow_seconds / main_seconds,
        },
        "config": config,
        "dataset_path": args.dataset,
        "details": all_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(
        f"Accuracy: {summary['accuracy']:.4f} ({summary['matches']}/{len(records)}); "
        f"added time: {output['timing']['added_time_fraction']:.4%}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
