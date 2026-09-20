"""Fixed Floodgate cost pairs only; never launches formal strength matches."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import cshogi

from train_nnue.match_protocol_pilot import AuditedEngine, dump, sha256
from train_nnue.match_protocol_qualification import runtime_manifest
from train_nnue.run_match import play_game

QUALIFIED = Path("docs/research/results/match-protocol-qualification-pass-20260920")
POOL = Path("dataset/floodgate2025/match-plan-v1")


def verify_inputs(run):
    expected = json.loads((run / "input-manifest.json").read_text())
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"input changed: {path}")
    groups = {}
    for name, count in (("cost8", 8), ("openings10000", 10000), ("reserve", 1324)):
        rows = [json.loads(line) for line in (POOL / f"{name}.jsonl").read_text().splitlines()]
        if len(rows) != count or len({r["game_hash"] for r in rows}) != count:
            raise ValueError(f"wrong size or duplicate game: {name}")
        if any(r["split"] != "match" or r["overlaps_known_old_position"] for r in rows):
            raise ValueError(f"invalid split/known overlap: {name}")
        groups[name] = rows
    all_rows = sum(groups.values(), [])
    if len({r["game_hash"] for r in all_rows}) != len(all_rows):
        raise ValueError("cost/formal/reserve game overlap")
    if len({" ".join(r["sfen"].split()[:3]) for r in all_rows}) != len(all_rows):
        raise ValueError("cost/formal/reserve position overlap")
    for row in groups["cost8"]:
        board = cshogi.Board(row["initial_sfen"])
        for token in row["history_usi"]:
            move = board.move_from_usi(token)
            if not move or not board.is_legal(move):
                raise ValueError("illegal cost source history")
            board.push(move)
        if board.sfen() != row["sfen"]:
            raise ValueError("cost history mismatch")
    return groups["cost8"]


def cost_summary(records):
    completed = [r for r in records if r["complete"]]
    if len(completed) != 16:
        return dict(complete_games=len(completed), chunk_pairs=None,
                    strength_inference="not measured; incomplete cost sample")
    pairs = [sum(r["wall_seconds"] for r in completed if r["opening_index"] == i) for i in range(8)]
    # Twofold slowest pair allowance, plus a 10-minute per-chunk reserve.
    conservative = 2 * max(pairs)
    chunk = min(10000, math.floor((21600 - 600) / conservative))
    return dict(complete_games=16, pair_seconds=pairs, mean_pair_seconds=sum(pairs)/8,
                max_pair_seconds=max(pairs), planning_pair_seconds=conservative,
                formal_sequential_hours=sum(pairs)/8*10000/3600,
                conservative_formal_hours=conservative*10000/3600,
                chunk_pairs=chunk, expected_chunks=math.ceil(10000/chunk) if chunk else None,
                estimate_limitation="8 cost pairs only; no measured parallel scaling or duration guarantee",
                strength_inference="not measured; cost scores must not select models or conditions")


def execute(run, budget):
    out = run / "artifacts" / (time.strftime("execution-%Y%m%dT%H%M%S") + f"-{os.getpid()}")
    out.mkdir(parents=True)
    dump(run / "artifacts/latest.json", dict(directory=str(out)))
    records, engines = [], []
    status, failure = "failed", "interrupted"
    deadline = time.monotonic() + budget
    try:
        openings = verify_inputs(run)
        config = json.loads((run / "protocol.json").read_text())
        dump(out / "manifest.json", runtime_manifest(list(json.loads((run / "input-manifest.json").read_text())) +
             [run / "input-manifest.json", run / "handoff.md"]))
        dump(out / "protocol.json", config)
        dump(out / "openings.json", openings)
        for model in ("candidate", "control"):
            log = f"/tmp/decision-cost-{out.name}-{model}.log"
            engine = AuditedEngine(config["binary"], config["options"][model], deadline, log)
            engines.append(engine)
            dump(out / f"engine-{model}.json", dict(options=engine.options, advertised=engine.advertised, log=log))
            maps = Path(f"/proc/{engine.process.pid}/maps").read_text()
            (out / f"maps-{model}.txt").write_text(maps)
            libraries = sorted({line.split()[-1] for line in maps.splitlines()
                                if line.split()[-1].startswith("/") and ".so" in line.split()[-1]})
            dump(out / f"libraries-{model}.json", {p: sha256(p) for p in libraries})
        for i, opening in enumerate(openings):
            for color in (0, 1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("cost deadline reached")
                row = dict(opening_index=i, game_hash=opening["game_hash"], candidate_color=color,
                           complete=False, trace={})
                records.append(row)
                dump(out / "progress.json", dict(opening_index=i, candidate_color=color,
                     complete_games=sum(r["complete"] for r in records)))
                started = time.monotonic()
                try:
                    first, second = engines if color == 0 else engines[::-1]
                    result, _ = play_game(first, second, {"nodes": config["nodes"]}, opening["sfen"],
                        max_moves=config["max_moves"], history=True, details=row["trace"],
                        initial_sfen=opening["initial_sfen"], history_usi=opening["history_usi"])
                    if any(s["nodes"] is None for s in row["trace"]["searches"]):
                        raise RuntimeError("missing actual nodes")
                    row.update(candidate_score=(1 + (result if color == 0 else -result))/2, complete=True)
                    row["memory"] = [e.rss_kib() for e in engines]
                except BaseException as error:
                    row["error"] = repr(error)
                    raise
                finally:
                    row["wall_seconds"] = time.monotonic() - started
                    dump(out / f"game-{len(records):03d}.json", row)
                    dump(out / "records.json", records)
        status, failure = "complete", None
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        for engine in engines:
            try:
                engine.quit()
            except Exception as error:
                status, failure = "failed", f"cleanup: {error!r}; previous={failure}"
        summary = cost_summary(records)
        summary.update(status=status, failure=failure)
        dump(out / "summary.json", summary)
        (run / "result-summary.md").write_text(
            f"# H-DECISION-REPLICATION cost only\n\nStatus: {status}\nFailure: {str(failure)[:2000]}\n"
            f"Complete games: {summary['complete_games']}/16\n"
            f"Proposed pairs per <=6h chunk: {summary['chunk_pairs']}\nArtifacts: {out}\n\n"
            "No strength inference or model selection. Formal 20,000 games were NOT launched.\n"
            "Timing includes game reset and search; engine startup/hash checks are extra.\n"
            "Review all game traces, protocol logs and summary.json before scheduling formal chunks.\n")
    if status != "complete":
        raise RuntimeError(failure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--budget-seconds", type=int, default=20400)
    args = parser.parse_args()
    if args.check_only:
        print(f"Verified inputs; {len(verify_inputs(args.run_dir))} cost openings; no engine started")
    else:
        execute(args.run_dir.resolve(), args.budget_seconds)


if __name__ == "__main__":
    main()
