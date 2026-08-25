#!/usr/bin/env python3
"""Measure fixed-blend search-utility stability across node horizons."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np

try:
    from scripts.collect_search_leaf_groups import EngineProcess, RECORD_BYTES, record_to_sfen
    from scripts.collect_search_utility_teachers import (
        onnxruntime_library_dir,
        reference_utilities,
        search_candidate,
        select_conservative_teacher,
    )
except ModuleNotFoundError:
    from collect_search_leaf_groups import EngineProcess, RECORD_BYTES, record_to_sfen
    from collect_search_utility_teachers import (
        onnxruntime_library_dir,
        reference_utilities,
        search_candidate,
        select_conservative_teacher,
    )


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman_with_ties(left, right):
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    if left_ranks.std() == 0.0 or right_ranks.std() == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def remove_logit_bias(biased_gate, bias):
    weights = np.asarray(biased_gate, dtype=np.float64) * np.exp(
        -np.asarray(bias, dtype=np.float64)
    )
    return weights / weights.sum()


def apply_logit_bias(base_gate, bias):
    weights = np.asarray(base_gate, dtype=np.float64) * np.exp(
        np.asarray(bias, dtype=np.float64)
    )
    return weights / weights.sum()


def bootstrap_mean_ci(values, iterations=10_000, seed=42):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        count = min(1000, iterations - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - margin, center + margin]


def pair_summary(details, left_nodes, right_nodes):
    correlations = []
    winner_overlap = []
    move_agreements = []
    selected_agreements = []
    changed_agreements = []
    teacher_l1 = []
    for detail in details:
        left = detail["horizons"][str(left_nodes)]
        right = detail["horizons"][str(right_nodes)]
        correlation = spearman_with_ties(left["utilities_cp"], right["utilities_cp"])
        if correlation is not None:
            correlations.append(correlation)
        left_utilities = np.asarray(left["utilities_cp"])
        right_utilities = np.asarray(right["utilities_cp"])
        left_winners = set(np.flatnonzero(left_utilities == left_utilities.max()).tolist())
        right_winners = set(np.flatnonzero(right_utilities == right_utilities.max()).tolist())
        winner_overlap.append(bool(left_winners & right_winners))
        move_agreements.extend(
            left_move == right_move
            for left_move, right_move in zip(left["moves"], right["moves"])
        )
        selected_agreements.append(left["selected_candidate"] == right["selected_candidate"])
        changed_agreements.append(left["teacher_changed"] == right["teacher_changed"])
        teacher_l1.append(
            float(
                np.abs(
                    np.asarray(left["teacher_gate"], dtype=np.float64)
                    - np.asarray(right["teacher_gate"], dtype=np.float64)
                ).sum()
            )
        )

    def fraction_summary(values):
        successes = int(sum(values))
        return {
            "count": successes,
            "total": len(values),
            "fraction": successes / len(values),
            "wilson_95": wilson_interval(successes, len(values)),
        }

    return {
        "left_nodes": left_nodes,
        "right_nodes": right_nodes,
        "informative_rank_roots": len(correlations),
        "informative_rank_fraction": len(correlations) / len(details),
        "spearman_mean": float(np.mean(correlations)) if correlations else None,
        "spearman_median": float(np.median(correlations)) if correlations else None,
        "spearman_mean_bootstrap_95": bootstrap_mean_ci(correlations),
        "oracle_winner_set_overlap": fraction_summary(winner_overlap),
        "candidate_move_agreement": fraction_summary(move_agreements),
        "selected_candidate_agreement": fraction_summary(selected_agreements),
        "teacher_changed_decision_agreement": fraction_summary(changed_agreements),
        "teacher_l1_mean": float(np.mean(teacher_l1)),
        "teacher_l1_median": float(np.median(teacher_l1)),
    }


def horizon_summary(details, nodes):
    rows = [detail["horizons"][str(nodes)] for detail in details]
    changed = [row["teacher_changed"] for row in rows]
    gains = np.asarray([row["oracle_gain_cp"] for row in rows], dtype=np.float64)
    changed_count = int(sum(changed))
    return {
        "nodes": nodes,
        "teacher_changed_count": changed_count,
        "teacher_changed_fraction": changed_count / len(rows),
        "teacher_changed_wilson_95": wilson_interval(changed_count, len(rows)),
        "oracle_gain_cp_median": float(np.median(gains)),
        "oracle_gain_cp_p90": float(np.quantile(gains, 0.9)),
        "oracle_gain_positive_fraction": float((gains > 0).mean()),
        "candidate_move_diversity_mean": float(
            np.mean([len(set(row["moves"])) for row in rows])
        ),
    }


def classify(primary, long_horizon):
    rank_ci = primary["spearman_mean_bootstrap_95"]
    overlap_ci = primary["oracle_winner_set_overlap"]["wilson_95"]
    changed_ci = long_horizon["teacher_changed_wilson_95"]
    enough_information = primary["informative_rank_fraction"] >= 0.10
    supported = (
        enough_information
        and rank_ci is not None
        and rank_ci[0] > 0.30
        and overlap_ci[0] > 0.40
        and changed_ci[0] > 0.05
    )
    rejected = (
        (rank_ci is not None and rank_ci[1] < 0.30)
        or overlap_ci[1] < 0.40
        or changed_ci[1] < 0.05
    )
    if supported:
        return "supported"
    if rejected:
        return "rejected"
    return "inconclusive"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots-file", type=Path, required=True)
    parser.add_argument("--candidate-engine", type=Path, required=True)
    parser.add_argument("--candidate-eval-dir", type=Path, required=True)
    parser.add_argument("--expert-blending-dir", type=Path, required=True)
    parser.add_argument("--reference-engine", type=Path, required=True)
    parser.add_argument("--reference-eval-dir", type=Path, required=True)
    parser.add_argument("--candidate-nodes", type=int, nargs="+", required=True)
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
    horizons = sorted(set(args.candidate_nodes))
    if len(horizons) < 2 or min(horizons + [args.reference_nodes, args.max_roots, args.workers]) <= 0:
        parser.error("at least two positive horizons and positive counts are required")
    if not 0.0 < args.logit_bias <= 20.0 or args.minimum_improvement_cp < 0.0:
        parser.error("invalid bias or improvement threshold")
    size = args.roots_file.stat().st_size
    root_count = size // RECORD_BYTES
    if size % RECORD_BYTES or args.start_root + args.max_roots > root_count:
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
            records.append((local_index, args.start_root + local_index, source.read(RECORD_BYTES)))

    progress = 0
    progress_lock = threading.Lock()
    started = time.monotonic()

    def process_chunk(chunk):
        nonlocal progress
        env = os.environ.copy()
        ort_lib = onnxruntime_library_dir(args.candidate_engine)
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
                by_horizon = {}
                all_moves = []
                stable_gates = None
                for horizon_index, nodes in enumerate(horizons):
                    moves, observed_gates = [], []
                    for bias in biases:
                        move, gate = search_candidate(
                            candidate,
                            sfen,
                            nodes,
                            bias,
                            require_gate=False,
                        )
                        moves.append(move)
                        observed_gates.append(gate)
                    if stable_gates is None:
                        observed_index = next(
                            (
                                index
                                for index, gate in enumerate(observed_gates[1:], start=1)
                                if gate is not None
                            ),
                            0 if observed_gates[0] is not None else None,
                        )
                        if observed_index is None:
                            raise ValueError(
                                f"no candidate emitted a gate: root={source_index}, sfen={sfen}"
                            )
                        base_gate = remove_logit_bias(
                            observed_gates[observed_index], biases[observed_index]
                        )
                        stable_gates = [
                            apply_logit_bias(base_gate, bias).astype(np.float32)
                            for bias in biases
                        ]
                    for candidate_index, gate in enumerate(observed_gates):
                        tolerance = 2e-3 if candidate_index == 0 else 2e-6
                        if gate is not None and not np.allclose(
                            gate, stable_gates[candidate_index], atol=tolerance
                        ):
                            max_error = float(
                                np.max(np.abs(gate - stable_gates[candidate_index]))
                            )
                            raise ValueError(
                                "candidate gate changed across horizons: "
                                f"root={source_index}, horizon={nodes}, "
                                f"candidate={candidate_index}, max_error={max_error}, "
                                f"observed={gate.tolist()}, "
                                f"expected={stable_gates[candidate_index].tolist()}"
                            )
                    by_horizon[nodes] = {"moves": moves, "gates": stable_gates}
                    all_moves.extend(moves)
                _, move_scores = reference_utilities(
                    reference, sfen, all_moves, args.reference_nodes
                )
                detail_horizons = {}
                for nodes in horizons:
                    row = by_horizon[nodes]
                    utilities = np.asarray([move_scores[move] for move in row["moves"]])
                    teacher, selected, gain = select_conservative_teacher(
                        row["gates"], utilities, args.minimum_improvement_cp
                    )
                    detail_horizons[str(nodes)] = {
                        "moves": row["moves"],
                        "utilities_cp": utilities.astype(int).tolist(),
                        "gates": np.asarray(row["gates"]).tolist(),
                        "teacher_gate": teacher.tolist(),
                        "selected_candidate": selected,
                        "oracle_gain_cp": gain,
                        "teacher_changed": not np.allclose(teacher, row["gates"][0], atol=1e-7),
                    }
                output.append((local_index, {
                    "source_root": source_index,
                    "sfen": sfen,
                    "unique_move_scores_cp": move_scores,
                    "horizons": detail_horizons,
                }))
                with progress_lock:
                    progress += 1
                    if progress % 4 == 0 or progress == args.max_roots:
                        elapsed = time.monotonic() - started
                        print(json.dumps({
                            "completed": progress, "total": args.max_roots,
                            "roots_per_second": progress / elapsed,
                        }), flush=True)
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
    details = [item[1] for item in collected]

    horizon_summaries = {str(nodes): horizon_summary(details, nodes) for nodes in horizons}
    pair_summaries = {}
    for left_index, left in enumerate(horizons):
        for right in horizons[left_index + 1 :]:
            pair_summaries[f"{left}_vs_{right}"] = pair_summary(details, left, right)
    primary_key = f"{horizons[0]}_vs_{horizons[-1]}"
    verdict = classify(pair_summaries[primary_key], horizon_summaries[str(horizons[-1])])
    summary = {
        "format": "search-utility-horizon-diagnostic-v1",
        "roots": args.max_roots,
        "start_root": args.start_root,
        "candidate_nodes": horizons,
        "reference_nodes": args.reference_nodes,
        "logit_bias": args.logit_bias,
        "minimum_improvement_cp": args.minimum_improvement_cp,
        "primary_comparison": primary_key,
        "pre_registered_verdict": verdict,
        "horizons": horizon_summaries,
        "pairs": pair_summaries,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("w") as target:
        for detail in details:
            target.write(json.dumps(detail, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
