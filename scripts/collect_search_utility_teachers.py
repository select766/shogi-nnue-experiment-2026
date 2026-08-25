#!/usr/bin/env python3
"""Build conservative router teachers from fixed-blend root searches.

For every root, the current gate and eight one-axis logit-bias candidates are
searched with the same node budget.  Independent fixed-NNUE restricted-move
searches score the unique candidate moves.  The current gate is retained unless
a candidate move improves the reference score by a configured margin.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import threading
import time

import numpy as np

try:
    from scripts.collect_search_leaf_groups import EngineProcess, RECORD_BYTES, record_to_sfen
    from scripts.relabel_root_grouped_search import parse_score_and_pv
except ModuleNotFoundError:
    from collect_search_leaf_groups import EngineProcess, RECORD_BYTES, record_to_sfen
    from relabel_root_grouped_search import parse_score_and_pv


GATE_RE = re.compile(r"^info string blending_weight=\[([^]]+)\]$")


def parse_gate(line, n_experts=8):
    match = GATE_RE.match(line)
    if not match:
        return None
    values = np.asarray(
        [float(token.strip()) for token in match.group(1).split(",")],
        dtype=np.float32,
    )
    if values.shape != (n_experts,) or not np.all(np.isfinite(values)):
        return None
    if np.any(values < 0.0) or not np.isclose(values.sum(), 1.0, atol=2e-5):
        return None
    return values / values.sum()


def select_conservative_teacher(gates, utilities, minimum_improvement):
    """Return teacher weights, with candidate zero as the current-router prior."""
    gates = np.asarray(gates, dtype=np.float32)
    utilities = np.asarray(utilities, dtype=np.float64)
    best = float(utilities.max())
    base = float(utilities[0])
    if best - base < minimum_improvement:
        return gates[0].copy(), 0, best - base
    winners = np.flatnonzero(utilities == best)
    teacher = gates[winners].mean(axis=0)
    teacher /= teacher.sum()
    return teacher, int(winners[0]), best - base


def search_candidate(engine, sfen, nodes, bias):
    encoded = ",".join(f"{value:.8g}" for value in bias)
    engine.send(f"setoption name ExpertBlendingGateLogitBias value {encoded}")
    engine.send(f"position sfen {sfen}")
    engine.send(f"go nodes {nodes}")
    gate = None
    bestmove = None
    while True:
        line = engine.read_until(lambda _: True)
        parsed_gate = parse_gate(line)
        if parsed_gate is not None:
            gate = parsed_gate
        if line.startswith("bestmove "):
            bestmove = line.split()[1]
            break
    if gate is None:
        raise ValueError("candidate search returned no blending_weight")
    return bestmove, gate


def reference_utilities(engine, sfen, moves, nodes):
    unique_moves = list(dict.fromkeys(moves))
    if any(move in {"resign", "win", "none"} for move in unique_moves):
        raise ValueError("candidate search returned a non-board move")
    scores = {}
    engine.send("setoption name MultiPV value 1")
    for move in unique_moves:
        # search_clear() is reached through isready, so every restricted move
        # receives the same node budget without a preceding move's TT entries.
        engine.send("isready")
        engine.read_until(lambda line: line == "readyok")
        engine.send(f"position sfen {sfen}")
        engine.send(f"go nodes {nodes} searchmoves {move}")
        latest = None
        while True:
            line = engine.read_until(lambda _: True)
            parsed = parse_score_and_pv(line)
            if parsed is not None:
                latest = parsed[0]
            if line.startswith("bestmove "):
                break
        if latest is None:
            raise ValueError(f"reference search returned no score for {move}")
        scores[move] = latest
    return np.asarray([scores[move] for move in moves], dtype=np.float64), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots-file", type=Path, required=True)
    parser.add_argument("--candidate-engine", type=Path, required=True)
    parser.add_argument("--candidate-eval-dir", type=Path, required=True)
    parser.add_argument("--expert-blending-dir", type=Path, required=True)
    parser.add_argument("--reference-engine", type=Path, required=True)
    parser.add_argument("--reference-eval-dir", type=Path, required=True)
    parser.add_argument("--candidate-nodes", type=int, required=True)
    parser.add_argument("--reference-nodes", type=int, required=True)
    parser.add_argument("--logit-bias", type=float, default=1.0)
    parser.add_argument("--minimum-improvement-cp", type=float, default=10.0)
    parser.add_argument("--start-root", type=int, default=0)
    parser.add_argument("--max-roots", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    args = parser.parse_args()
    if min(args.candidate_nodes, args.reference_nodes, args.max_roots, args.workers) <= 0:
        parser.error("node/root/worker counts must be positive")
    if not 0.0 < args.logit_bias <= 20.0 or args.minimum_improvement_cp < 0.0:
        parser.error("invalid bias or improvement threshold")
    root_count = args.roots_file.stat().st_size // RECORD_BYTES
    stop = args.start_root + args.max_roots
    if args.roots_file.stat().st_size % RECORD_BYTES or stop > root_count:
        parser.error("requested root range is outside an aligned roots file")

    biases = [np.zeros(8, dtype=np.float32)]
    for expert in range(8):
        bias = np.zeros(8, dtype=np.float32)
        bias[expert] = args.logit_bias
        biases.append(bias)

    records = []
    with args.roots_file.open("rb") as source:
        source.seek(args.start_root * RECORD_BYTES)
        for local_index in range(args.max_roots):
            record = source.read(RECORD_BYTES)
            records.append((local_index, args.start_root + local_index, record))

    progress = 0
    progress_lock = threading.Lock()
    started = time.monotonic()

    def process_chunk(chunk):
        nonlocal progress
        env = os.environ.copy()
        ort_lib = (
            args.candidate_engine.resolve().parent.parent
            / "extra/onnxruntime/linux/current/lib"
        )
        env["LD_LIBRARY_PATH"] = str(ort_lib) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        candidate = EngineProcess([str(args.candidate_engine.resolve())], env=env)
        reference = EngineProcess([str(args.reference_engine.resolve())])
        candidate.initialize([
            ("Threads", 1), ("USI_Hash", args.hash_mb),
            ("EvalDir", args.candidate_eval_dir.resolve()),
            ("ExpertBlendingDir", args.expert_blending_dir.resolve()),
            ("ClearTTOnDynamicWeights", "true"), ("USI_OwnBook", "false"),
        ])
        reference.initialize([
            ("Threads", 1), ("USI_Hash", args.hash_mb),
            ("EvalDir", args.reference_eval_dir.resolve()), ("USI_OwnBook", "false"),
        ])
        import cshogi
        board = cshogi.Board()
        psfen = np.zeros(1, dtype=cshogi.PackedSfen)
        output = []
        try:
            for local_index, source_index, record in chunk:
                sfen = record_to_sfen(record, board, psfen)
                moves, gates = [], []
                for bias in biases:
                    move, gate = search_candidate(
                        candidate, sfen, args.candidate_nodes, bias
                    )
                    moves.append(move)
                    gates.append(gate)
                utilities, move_scores = reference_utilities(
                    reference, sfen, moves, args.reference_nodes
                )
                teacher, selected, gain = select_conservative_teacher(
                    gates, utilities, args.minimum_improvement_cp
                )
                regrets = (utilities.max() - utilities[1:]) / 361.0
                cache = np.concatenate([regrets.astype(np.float32), teacher])
                output.append((local_index, cache, {
                    "source_root": source_index,
                    "sfen": sfen,
                    "moves": moves,
                    "utilities_cp": utilities.astype(int).tolist(),
                    "unique_move_scores_cp": move_scores,
                    "gates": np.asarray(gates).tolist(),
                    "selected_candidate": selected,
                    "oracle_gain_cp": gain,
                    "teacher_changed": not np.allclose(teacher, gates[0], atol=1e-7),
                }))
                with progress_lock:
                    progress += 1
                    if progress % 10 == 0 or progress == args.max_roots:
                        elapsed = time.monotonic() - started
                        print(
                            json.dumps({
                                "completed": progress,
                                "total": args.max_roots,
                                "roots_per_second": progress / elapsed,
                            }),
                            flush=True,
                        )
        finally:
            candidate.close()
            reference.close()
        return output

    chunks = [[] for _ in range(args.workers)]
    for item in records:
        chunks[item[0] % args.workers].append(item)
    collected = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks if chunk]
        for future in as_completed(futures):
            collected.extend(future.result())
    collected.sort(key=lambda item: item[0])
    cache = np.stack([item[1] for item in collected])
    details = [item[2] for item in collected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as target:
        np.save(target, cache)
    os.replace(temporary, args.output)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("w") as target:
        for detail in details:
            target.write(json.dumps(detail, ensure_ascii=False) + "\n")
    gains = np.asarray([item["oracle_gain_cp"] for item in details])
    changed = np.asarray([item["teacher_changed"] for item in details])
    summary = {
        "format": "search-utility-teacher-v1", "roots": args.max_roots,
        "start_root": args.start_root, "candidate_nodes": args.candidate_nodes,
        "reference_nodes": args.reference_nodes, "logit_bias": args.logit_bias,
        "minimum_improvement_cp": args.minimum_improvement_cp,
        "candidate_definition": "current gate plus +logit_bias toward each of 8 experts",
        "teacher_definition": "retain current gate unless independent restricted-move oracle improves by threshold; otherwise mean gate of tied best candidates",
        "teacher_changed_fraction": float(changed.mean()),
        "oracle_gain_cp_mean": float(gains.mean()),
        "oracle_gain_cp_positive_fraction": float((gains > 0).mean()),
        "candidate_move_diversity_mean": float(np.mean([len(set(item["moves"])) for item in details])),
        "elapsed_seconds": time.monotonic() - started,
    }
    with args.output.with_suffix(".json").open("w") as target:
        json.dump(summary, target, indent=2)
        target.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
