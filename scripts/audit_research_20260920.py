#!/usr/bin/env python3
"""Reanalyse saved matches without starting engines or changing old results."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import cshogi

from train_nnue.run_match import play_game


def identity(sfen):
    """Position identity excludes the SFEN move counter."""
    fields = sfen.split()
    if len(fields) != 4:
        raise ValueError("expected four SFEN fields")
    return " ".join(fields[:3])


def paired_summary(details):
    pairs = defaultdict(dict)
    scores = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    for row in details:
        key, color = identity(row["sfen"]), row["engine1_color"]
        if color not in {"black", "white"} or color in pairs[key]:
            raise ValueError("invalid or duplicate SFEN/color")
        pairs[key][color] = scores[row["engine1_result"]]
    if len(pairs) < 2 or any(set(v) != {"black", "white"} for v in pairs.values()):
        raise ValueError("at least two complete color pairs required")
    values = [sum(v.values()) / 2 for v in pairs.values()]
    score = statistics.mean(values)
    if not 0 < score < 1:
        raise ValueError("boundary score has no finite delta-method Elo interval")
    elo = 400 * math.log10(score / (1 - score))
    # Each start position is one observation, including both reversed colors.
    se_score = statistics.stdev(values) / math.sqrt(len(values))
    se_elo = 400 / (math.log(10) * score * (1 - score)) * se_score
    return {
        "games": len(details), "opening_pairs": len(values),
        "outcomes": dict(Counter(r["engine1_result"] for r in details)),
        "pair_total_score_counts": dict(sorted(Counter(2 * v for v in values).items())),
        "score_rate": score, "elo": elo, "pair_elo_se": se_elo,
        "pair_elo_95": [elo - 1.959963984540054 * se_elo,
                        elo + 1.959963984540054 * se_elo],
    }


def history_probe():
    """Probe the current checkout; saved historical probes remain immutable."""
    calls = []

    class SpyEngine:
        def isready(self):
            pass

        def usinewgame(self):
            pass

        def position(self, *, sfen, moves=None):
            self.sfen = sfen + (" moves " + " ".join(moves) if moves else "")
            self.board = cshogi.Board(sfen.removeprefix("sfen "))
            for move in moves or []:
                self.board.push_usi(move)

        def go(self, **kwargs):
            calls.append(self.sfen)
            return cshogi.move_to_usi(next(iter(self.board.legal_moves))), None

    play_game(SpyEngine(), SpyEngine(), {"nodes": 1}, max_moves=6)
    return {"plies": len(calls), "search_positions": calls,
            "positions_with_move_history": sum(" moves " in v for v in calls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}

    def read_json(path):
        path = Path(path)
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def read_jsonl(path):
        path = Path(path)
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    runs = {n: read_json(f"results/match_decision_aligned_{n}.json")
            for n in (4000, 12000, 14000)}
    summaries = {}
    for n, run in runs.items():
        summary = paired_summary(run["details"])
        assert summary["games"] == run["games"] == n
        for label, field in (("win", "wins"), ("loss", "losses"), ("draw", "draws")):
            assert summary["outcomes"].get(label, 0) == run[field]
        summary["legacy_game_independent_elo_95"] = run["elo_95"]
        summaries[str(n)] = summary
    # Verify nesting by position/color/outcome rather than relying on file order.
    def records(run):
        return {(identity(r["sfen"]), r["engine1_color"], r["engine1_result"])
                for r in run["details"]}
    assert records(runs[4000]) < records(runs[12000]) < records(runs[14000])
    first_keys = {identity(r["sfen"]) for r in runs[4000]["details"]}
    middle_keys = {identity(r["sfen"]) for r in runs[12000]["details"]}
    for name, full, excluded in (("extension_8000", 12000, first_keys),
                                  ("sensitivity_2000", 14000, middle_keys),
                                  ("all_new_10000", 14000, first_keys)):
        summaries[name] = paired_summary([
            r for r in runs[full]["details"] if identity(r["sfen"]) not in excluded
        ])
    teacher_rows = read_jsonl("tmp/search_utility_v1/pilot/details.jsonl")
    teacher_keys = {identity(r["sfen"]) for r in teacher_rows}
    opening_keys = {identity(r["sfen"]) for r in runs[14000]["details"]}
    train_keys = {identity(r["sfen"]) for r in teacher_rows[:1600]}
    val_keys = {identity(r["sfen"]) for r in teacher_rows[1600:]}
    summaries["14000_excluding_teacher_overlap"] = paired_summary([
        r for r in runs[14000]["details"] if identity(r["sfen"]) not in teacher_keys
    ])
    # With no game-boundary reset, chunk dependence is a useful stress analysis.
    chunks = defaultdict(list)
    for row in runs[14000]["details"]:
        chunks[row["chunk"]].append({"win": 1, "draw": 0.5, "loss": 0}[row["engine1_result"]])
    assert len(chunks) == 140 and {len(v) for v in chunks.values()} == {100}
    chunk_scores = [statistics.mean(v) for v in chunks.values()]
    p = statistics.mean(chunk_scores)
    chunk_se = statistics.stdev(chunk_scores) / math.sqrt(len(chunks)) * 400 / (math.log(10) * p * (1-p))
    elo = summaries["14000"]["elo"]
    evaluation = {}
    for split in ("validation", "test"):
        keys = {identity(r["sfen"]) for r in read_jsonl(f"data/accuracy_eval_10k/{split}.jsonl")}
        evaluation[split] = {"unique_positions": len(keys),
                             "teacher_overlap": len(keys & teacher_keys)}
    result = {
        "format": "research-audit-20260920-v1",
        "method": "opening-pair sample variance, delta-method normal 95% interval; no selection correction",
        "limitations": "SFEN disjointness does not establish source-game independence; no full pretraining overlap audit",
        "matches": summaries,
        "chunk_sensitivity_14000": {
            "chunks": len(chunks), "games_per_chunk": 100,
            "normal_95": [elo - 1.959963984540054 * chunk_se,
                          elo + 1.959963984540054 * chunk_se],
            "note": "chunk clusters address within-process dependence, not source-game dependence",
        },
        "teacher": {"rows": len(teacher_rows), "unique_positions": len(teacher_keys),
                    "train_validation_overlap": len(train_keys & val_keys),
                    "train_validation_overlap_positions": sorted(train_keys & val_keys),
                    "match_opening_overlap": len(teacher_keys & opening_keys),
                    "match_opening_overlap_positions": sorted(teacher_keys & opening_keys),
                    "evaluation": evaluation,
                    "generation": read_json("tmp/search_utility_v1/pilot/teacher.json")},
        "history_probe": history_probe(),
        "protocol": {"engine1": runs[14000]["engine1"],
                     "engine2": runs[14000]["engine2"],
                     "clear_hash_each_move": runs[14000].get("clear_hash_each_move", False)},
        "source_sha256": sources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
