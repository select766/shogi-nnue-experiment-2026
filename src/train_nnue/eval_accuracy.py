"""Evaluate best-move accuracy of a shogi engine against a JSONL dataset."""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from cshogi.usi import Engine


def resolve_path(path, project_root):
    """Resolve a path relative to project_root if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


def load_config(config_path):
    """Load JSON config file."""
    with open(config_path) as f:
        return json.load(f)


def load_dataset(dataset_path):
    """Load JSONL dataset."""
    records = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def create_engine(engine_path, engine_options):
    """Create and initialize an Engine instance."""
    engine = Engine(engine_path)
    for key, value in engine_options.items():
        engine.setoption(key, value)
    engine.isready()
    return engine


def evaluate_position(engine, sfen, go_params):
    """Evaluate a single position and return the engine's bestmove."""
    engine.position(sfen=f"sfen {sfen}")
    bestmove, _ = engine.go(**go_params)
    return bestmove


def worker_fn(positions, engine_path, engine_options, go_params, progress_callback):
    """Worker function that processes a list of positions with its own engine."""
    engine = create_engine(engine_path, engine_options)
    results = []
    try:
        for idx, record in positions:
            bestmove = evaluate_position(engine, record["sfen"], go_params)
            match = bestmove == record["bestmove"]
            results.append({
                "index": idx,
                "sfen": record["sfen"],
                "expected": record["bestmove"],
                "actual": bestmove,
                "match": match,
            })
            progress_callback()
    finally:
        engine.quit()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate best-move accuracy of a shogi engine"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL dataset file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON results file",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root for resolving relative paths in config (default: current directory)",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)

    # Load config
    config = load_config(args.config)
    engine_path = resolve_path(config["engine_path"], project_root)
    engine_options = dict(config.get("engine_options", {}))
    go_params = config.get("go_params", {"nodes": 1000000})
    num_workers = config.get("num_workers", 4)

    # Resolve EvalDir if present
    if "EvalDir" in engine_options:
        engine_options["EvalDir"] = resolve_path(
            engine_options["EvalDir"], project_root
        )

    print(f"Engine: {engine_path}", file=sys.stderr)
    print(f"Options: {engine_options}", file=sys.stderr)
    print(f"Go params: {go_params}", file=sys.stderr)
    print(f"Workers: {num_workers}", file=sys.stderr)

    # Load dataset
    records = load_dataset(args.dataset)
    total = len(records)
    print(f"Dataset: {total} positions", file=sys.stderr)

    # Split positions across workers
    indexed_records = list(enumerate(records))
    chunks = [[] for _ in range(num_workers)]
    for i, rec in enumerate(indexed_records):
        chunks[i % num_workers].append(rec)

    # Progress tracking
    progress_lock = threading.Lock()
    progress_count = [0]

    def progress_callback():
        with progress_lock:
            progress_count[0] += 1
            count = progress_count[0]
        if count % 100 == 0 or count == total:
            print(f"Progress: {count}/{total}", file=sys.stderr)

    # Run workers
    all_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for chunk in chunks:
            if chunk:
                futures.append(
                    executor.submit(
                        worker_fn,
                        chunk,
                        engine_path,
                        engine_options,
                        go_params,
                        progress_callback,
                    )
                )

        for future in as_completed(futures):
            all_results.extend(future.result())

    # Sort by original index
    all_results.sort(key=lambda x: x["index"])

    # Compute accuracy
    matches = sum(1 for r in all_results if r["match"])
    accuracy = matches / total if total > 0 else 0.0

    output = {
        "accuracy": accuracy,
        "matches": matches,
        "total": total,
        "config": config,
        "dataset_path": args.dataset,
        "details": all_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {accuracy:.4f} ({matches}/{total})", file=sys.stderr)
    print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
