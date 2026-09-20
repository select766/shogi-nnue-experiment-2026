"""Fixed daily growth measurements. Engines run only in the external queue phase."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import importlib.metadata
from pathlib import Path
import random
import statistics
import subprocess
import threading
import uuid

import cshogi

from train_nnue.accuracy_statistics import accuracy_summary
from train_nnue.eval_accuracy import create_engine, resolve_engine_option_paths
from train_nnue.research_loop import ROOT, ID, now, read_json, write_json


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(sfen):
    return " ".join(sfen.split()[:3])


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fingerprint(root, paths):
    return {path: digest(root / path) for path in sorted(set(paths))}


def verify(root, hashes):
    for path, expected in hashes.items():
        if digest(root / path) != expected:
            raise ValueError(f"pinned asset changed: {path}; use a new series/model ID")


def runtime():
    return {"cshogi": importlib.metadata.version("cshogi")}


def pin_champion(root, directory, descriptor):
    champion = read_json(root / descriptor)
    if not ID.fullmatch(champion["id"]):
        raise ValueError("invalid champion ID")
    if not (root / champion["evidence"]).is_file() or not champion["selection_note"].strip():
        raise ValueError("champion needs selection evidence and note")
    required = {champion["engine_path"], champion["checkpoint"], champion["evidence"]}
    options = champion["engine_options"]
    required.add(str(Path(options["EvalDir"]) / "nn.bin"))
    if options.get("ExpertBlendingDir"):
        required.update(str(Path(options["ExpertBlendingDir"]) / name)
                        for name in ("backbone.onnx", "head.bin"))
    if options.get("Threads") != 1:
        raise ValueError("daily champion must use Threads=1")
    payload = {"descriptor": champion, "sha256": fingerprint(root, required | set(champion["assets"]))}
    path = directory / "champions" / f"{champion['id']}.json"
    if path.exists():
        if read_json(path) != payload:
            raise ValueError("champion ID is immutable; give a changed model a new ID")
    else:
        write_json(path, payload)
    return payload


def initialize(root, state_dir, config_path):
    config = read_json(root / config_path)
    if not ID.fullmatch(config["series"]):
        raise ValueError("invalid benchmark series ID")
    for key in ("accuracy_positions", "accuracy_nodes", "opening_count", "match_nodes", "max_moves", "workers"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"invalid {key}")
    if config["workers"] > 4:
        raise ValueError("daily workers capped at 4")
    directory = state_dir / "growth" / config["series"]
    directory.mkdir(parents=True, exist_ok=True)
    protocol = directory / "protocol.json"
    if protocol.exists():
        pinned = read_json(protocol)
        if pinned["config"] != config:
            raise ValueError("daily protocol changed; choose a new series ID")
        verify(root, pinned["sha256"])
        verify(directory, pinned["dataset_sha256"])
        if pinned["runtime"] != runtime():
            raise ValueError("evaluation dependency changed; use a new series")
        pin_champion(root, directory, config["champion"])
        return directory
    accuracy = records(root / config["accuracy_dataset"])
    if len(accuracy) != config["accuracy_positions"] or len({identity(r["sfen"]) for r in accuracy}) != len(accuracy):
        raise ValueError("accuracy dataset size/uniqueness mismatch")
    eligible = {}
    excluded = {identity(r["sfen"]) for r in accuracy}
    for row in records(root / config["opening_source"]):
        key = identity(row["sfen"])
        if key not in excluded and abs(row["eval"]) <= config["max_abs_eval"] and not cshogi.Board(row["sfen"]).is_game_over():
            eligible.setdefault(key, row)
    openings = list(eligible.values())
    random.Random(config["opening_seed"]).shuffle(openings)
    if len(openings) < config["opening_count"]:
        raise ValueError("not enough disjoint, eligible opening positions")
    for filename, rows in (("accuracy.jsonl", accuracy), ("openings.jsonl", openings[:config["opening_count"]])):
        (directory / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    baseline = config["baseline"]
    if baseline["engine_options"].get("Threads") != 1:
        raise ValueError("baseline Threads must be 1")
    files = set(baseline["assets"]) | {baseline["engine_path"],
             str(Path(baseline["engine_options"]["EvalDir"]) / "nn.bin"),
             config["accuracy_dataset"], config["opening_source"],
             "src/train_nnue/daily_benchmark.py", "src/train_nnue/accuracy_statistics.py",
             "src/train_nnue/eval_accuracy.py"}
    pin_champion(root, directory, config["champion"])
    write_json(protocol, {"config": config, "created_at": now(),
                         "runtime": runtime(),
                         "sha256": fingerprint(root, files),
                         "dataset_sha256": fingerprint(directory, ["accuracy.jsonl", "openings.jsonl"])})
    render(root, directory)
    return directory


def due(state, seconds, clock=None):
    clock = clock or datetime.now(timezone.utc)
    jobs = [j for j in state["jobs"] if j.get("kind") == "daily_benchmark"]
    if any(j["status"] in {"queued", "blocked"} for j in jobs):
        return False
    if not jobs:
        return True
    latest = max(datetime.fromisoformat(j["created_at"]) for j in jobs)
    return (clock - latest).total_seconds() >= seconds


def schedule(root, store, settings, force=False):
    # Call only under the worker lock (or administrator's worker/workspace locks).
    state = store.snapshot()
    if settings["interval_seconds"] <= 0 or settings["timeout_seconds"] <= 0:
        raise ValueError("daily interval/timeout must be positive")
    if not force and not due(state, settings["interval_seconds"]):
        return None
    if any(j.get("kind") == "daily_benchmark" and j["status"] in {"queued", "blocked"} for j in state["jobs"]):
        raise ValueError("a daily benchmark is already pending")
    directory = initialize(root, store.directory, Path(settings["config"]))
    config = read_json(directory / "protocol.json")["config"]
    champion = pin_champion(root, directory, config["champion"])
    key = "daily-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = store.directory / "experiments" / key / "attempt-001"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "benchmark.json", {"series_dir": str(directory), "champion": champion,
                                          "protocol_sha256": digest(directory / "protocol.json")})
    job = {"id": key, "kind": "daily_benchmark", "hypothesis": "", "task": "Fixed daily growth measurement",
           "series": config["series"],
           "depends_on": [], "status": "queued", "phase": "execute", "attempts": [str(run_dir)],
           "created_at": now(), "message": "", "timeout_seconds": settings["timeout_seconds"]}
    with store.edit() as state:
        state["jobs"].append(job)
        state.pop("benchmark_error", None)
    return key


def new_engine(root, descriptor):
    return create_engine(str(root / descriptor["engine_path"]),
                         resolve_engine_option_paths(descriptor["engine_options"], str(root)))


def game(black, white, sfen, nodes, max_moves):
    """History-aware daily protocol, independent of the legacy research runner."""
    board = cshogi.Board(sfen)
    moves = []
    for engine in (black, white):
        engine.isready()
        engine.usinewgame()
    for _ in range(max_moves):
        side = board.turn
        engine = (black, white)[side]
        position = f"sfen {sfen}" + (" moves " + " ".join(moves) if moves else "")
        engine.position(sfen=position)
        # Both sides clear TT; no old dynamic evaluations or inter-game history.
        engine.setoption("Clear Hash", "")
        bestmove, _ = engine.go(nodes=nodes)
        if bestmove == "resign":
            return (1 if side == 0 else 0), moves, "resign"
        if bestmove == "win":
            if not board.is_nyugyoku():
                raise ValueError("invalid entering-king declaration")
            return side, moves, "declaration"
        if not bestmove:
            raise ValueError("missing bestmove")
        move = board.move_from_usi(bestmove)
        if not move or not board.is_legal(move):
            raise ValueError(f"illegal move: {bestmove}")
        board.push(move)
        moves.append(bestmove)
        repetition = board.is_draw()
        if repetition == cshogi.REPETITION_DRAW:
            return None, moves, "repetition"
        if repetition == cshogi.REPETITION_WIN:
            return board.turn, moves, "perpetual-check"
        if repetition == cshogi.REPETITION_LOSE:
            return 1 - board.turn, moves, "perpetual-check"
        if board.is_game_over():
            return 1 - board.turn, moves, "game-over"
    return None, moves, "max-moves"


def measure(root, run_dir, protocol, champion):
    config = protocol["config"]
    series = Path(read_json(run_dir / "benchmark.json")["series_dir"])
    accuracy_rows, openings = records(series / "accuracy.jsonl"), records(series / "openings.jsonl")
    failed = threading.Event()

    def worker(index):
        candidate = opponent = None
        accuracy, matches = [], []
        try:
            candidate = new_engine(root, champion)
            for i in range(index, len(accuracy_rows), config["workers"]):
                if failed.is_set():
                    raise RuntimeError("another benchmark worker failed")
                row = accuracy_rows[i]
                candidate.setoption("Clear Hash", "")
                candidate.position(sfen="sfen " + row["sfen"])
                move, _ = candidate.go(nodes=config["accuracy_nodes"])
                board = cshogi.Board(row["sfen"])
                if not move or (move not in {"resign", "win"} and not board.is_legal(board.move_from_usi(move))):
                    raise ValueError("non-legal bestmove in accuracy evaluation")
                accuracy.append({"index": i, "sfen": row["sfen"], "actual": move,
                                 "expected": row["bestmove"], "match": move == row["bestmove"]})
                if len(accuracy) % 100 == 0:
                    write_json(run_dir / f"progress-{index}.json", {"stage": "accuracy", "accuracy": len(accuracy), "games": 0})
            opponent = new_engine(root, config["baseline"])
            for i in range(index, len(openings), config["workers"]):
                for color in (0, 1):
                    if failed.is_set():
                        raise RuntimeError("another benchmark worker failed")
                    winner, moves, reason = game(candidate if color == 0 else opponent,
                                                 opponent if color == 0 else candidate,
                                                 openings[i]["sfen"], config["match_nodes"], config["max_moves"])
                    outcome = "draw" if winner is None else "win" if winner == color else "loss"
                    matches.append({"opening": i, "sfen": openings[i]["sfen"], "color": color,
                                    "outcome": outcome, "moves": moves, "reason": reason})
                write_json(run_dir / f"progress-{index}.json", {"accuracy": len(accuracy), "games": len(matches)})
        except Exception:
            failed.set()
            raise
        finally:
            if candidate is not None:
                candidate.quit()
            if opponent is not None:
                opponent.quit()
            write_json(run_dir / f"worker-{index}.json", {"accuracy": accuracy, "matches": matches})
        return accuracy, matches

    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        pieces = list(pool.map(worker, range(config["workers"])))
    return ([v for part in pieces for v in part[0]], [v for part in pieces for v in part[1]])


def pair_metric(matches, score):
    pairs = {}
    for row in matches:
        colors = pairs.setdefault(identity(row["sfen"]), {})
        if row["color"] not in (0, 1) or row["color"] in colors:
            raise ValueError("duplicate/invalid color pair")
        colors[row["color"]] = score[row["outcome"]]
    if len(pairs) < 2 or any(set(v) != {0, 1} for v in pairs.values()):
        raise ValueError("incomplete match pairs")
    values = [sum(v.values()) / 2 for v in pairs.values()]
    mean = statistics.mean(values)
    radius = 1.959963984540054 * statistics.stdev(values) / math.sqrt(len(values))
    return {"value": mean, "lower": max(0, mean-radius), "upper": min(1, mean+radius)}


def summarize(accuracy, matches, protocol):
    cfg = protocol["config"]
    if len(accuracy) != cfg["accuracy_positions"] or {v["index"] for v in accuracy} != set(range(len(accuracy))):
        raise ValueError("incomplete accuracy measurement")
    if len(matches) != cfg["opening_count"] * 2 or {v["opening"] for v in matches} != set(range(cfg["opening_count"])):
        raise ValueError("incomplete matches")
    return {"accuracy": accuracy_summary([v["match"] for v in accuracy]),
            "games": len(matches), "wins": sum(v["outcome"] == "win" for v in matches),
            "losses": sum(v["outcome"] == "loss" for v in matches),
            "draws": sum(v["outcome"] == "draw" for v in matches),
            "win_rate": pair_metric(matches, {"win": 1, "draw": 0, "loss": 0}),
            "score_rate": pair_metric(matches, {"win": 1, "draw": .5, "loss": 0})}


def render(root, directory):
    subprocess.run([str(root / "scripts/nnue_python.sh"), "-m", "train_nnue.plot_growth",
                    "--series-dir", str(directory)], cwd=root, check=True, timeout=300)


def execute(root, run_dir):
    request = read_json(run_dir / "benchmark.json")
    directory = Path(request["series_dir"])
    protocol = read_json(directory / "protocol.json")
    if digest(directory / "protocol.json") != request["protocol_sha256"]:
        raise ValueError("scheduled protocol manifest changed")
    verify(root, protocol["sha256"])
    if protocol["runtime"] != runtime():
        raise ValueError("evaluation dependency changed; use a new series")
    verify(directory, protocol["dataset_sha256"])
    verify(root, request["champion"]["sha256"])
    result_path = directory / "measurements" / f"{run_dir.parent.name}.json"
    # Retry after a completed durable measurement only rebuilds the presentation.
    if not result_path.exists():
        started = now()
        accuracy, matches = measure(root, run_dir, protocol, request["champion"]["descriptor"])
        metrics = summarize(accuracy, matches, protocol)
        verify(root, protocol["sha256"])
        verify(root, request["champion"]["sha256"])
        verify(directory, protocol["dataset_sha256"])
        if digest(directory / "protocol.json") != request["protocol_sha256"]:
            raise ValueError("protocol manifest changed during measurement")
        write_json(run_dir / "accuracy.json", accuracy)
        write_json(run_dir / "matches.json", matches)
        write_json(result_path, {"started_at": started, "finished_at": now(),
                                "champion": request["champion"], "metrics": metrics,
                                "protocol_sha256": digest(directory / "protocol.json"),
                                "run_dir": str(run_dir)})
    render(root, directory)
    write_json(run_dir / "benchmark-complete.json", {"measurement": str(result_path),
                                                    "dashboard": str(directory / "growth.html")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(ROOT, args.run_dir)


if __name__ == "__main__":
    main()
