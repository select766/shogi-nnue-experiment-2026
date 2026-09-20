"""Bounded 2x2 protocol diagnostic. No training, selection, or strength claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import select
import subprocess
import time

import cshogi

from train_nnue.run_match import play_game


class BudgetExpired(TimeoutError):
    pass


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def generated_openings(count=16, seed=20260920):
    """Synthetic legal walks, used only for protocol diagnostics (not strength)."""
    rng = random.Random(seed)
    result, seen = [], set()
    while len(result) < count:
        board = cshogi.Board()
        moves = []
        for _ in range(24):
            legal = sorted(board.legal_moves, key=cshogi.move_to_usi)
            if not legal:
                break
            move = rng.choice(legal)
            moves.append(cshogi.move_to_usi(move))
            board.push(move)
            if board.is_draw() != cshogi.NOT_REPETITION:
                break
        key = " ".join(board.sfen().split()[:3])
        if len(moves) != 24 or board.is_game_over() or board.is_draw() or key in seen:
            continue
        seen.add(key)
        result.append(dict(index=len(result), sfen=board.sfen(), source="synthetic-legal-walk",
                           seed=seed, source_start=cshogi.STARTING_SFEN, source_moves=moves))
    return result


class AuditedEngine:
    """Synchronous USI with bounded reads and a durable command/response log."""
    def __init__(self, binary, options, deadline, log, stop_event=None):
        self.stop_event = stop_event
        self.deadline = deadline
        self.log = Path(log).open("w", buffering=1)
        self.buffer = b""
        self.options = dict(options)
        env = dict(os.environ)
        lib = str(Path("YaneuraOu/extra/onnxruntime/linux/current/lib").resolve())
        env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")
        self.process = subprocess.Popen([str(Path(binary).resolve())], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        try:
            self.send("usi")
            self.advertised = []
            while (line := self.read()) != "usiok":
                self.advertised.append(line)
            for name, value in options.items():
                if not any(line.startswith(f"option name {name} type ") for line in self.advertised):
                    raise RuntimeError(f"required USI option missing: {name}")
                self.setoption(name, value)
            self.isready()
        except BaseException:
            self.quit()
            raise

    def send(self, command):
        self.log.write(f"> {command}\n")
        self.process.stdin.write((command + "\n").encode())
        self.process.stdin.flush()

    def read(self):
        local_deadline = min(self.deadline, time.monotonic() + 120)
        while b"\n" not in self.buffer:
            if self.stop_event is not None and self.stop_event.is_set():
                raise InterruptedError("match worker cancelled")
            remaining = local_deadline - time.monotonic()
            if remaining <= 0:
                if time.monotonic() >= self.deadline:
                    raise BudgetExpired("50-minute compute deadline")
                raise TimeoutError("USI response exceeded 120 seconds")
            if not select.select([self.process.stdout], [], [], min(remaining, 0.2))[0]:
                continue
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"USI engine EOF: {self.process.poll()}")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        text = line.decode(errors="replace").strip()
        self.log.write(f"< {text}\n")
        return text

    def setoption(self, name, value):
        self.options[name] = value
        self.send(f"setoption name {name} value {value}")

    def isready(self):
        self.send("isready")
        while self.read() != "readyok":
            pass

    def usinewgame(self):
        self.send("usinewgame")

    def position(self, *, sfen, moves=None):
        self.send("position " + sfen + (" moves " + " ".join(moves) if moves else ""))

    def go(self, *, nodes, listener, searchmoves=None):
        self.send(f"go nodes {nodes}" + (" searchmoves " + " ".join(searchmoves) if searchmoves else ""))
        cache = []
        blended = 0
        while True:
            line = self.read()
            if line.startswith("info string dynamic_weight_cache "):
                cache.append(line)
            if line.startswith("info string blending_weight="):
                blended += 1
            if "failed" in line.lower() or "error" in line.lower():
                raise RuntimeError(f"engine diagnostic: {line}")
            listener(line)
            if line.startswith("bestmove "):
                expected = int(self.options["ClearTTOnDynamicWeights"] == "true")
                if len(cache) != 1 or f"clear={expected} " not in cache[0] or blended != 1:
                    raise RuntimeError(f"dynamic weight/cache evidence missing: {cache}, blend={blended}")
                return line.split()[1], None

    def rss_kib(self):
        return [line.strip() for line in Path(f"/proc/{self.process.pid}/status").read_text().splitlines()
                if line.startswith(("VmRSS:", "VmHWM:"))]

    def quit(self):
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


def common_complete(records):
    expected = {(history, clear, color) for history in (False, True)
                for clear in (False, True) for color in (0, 1)}
    indices = sorted({r["opening_index"] for r in records if r["stage"] == "factorial"})
    return [index for index in indices
            if {(r["history"], r["clear"], r["candidate_color"]) for r in records
                if r["stage"] == "factorial" and r["opening_index"] == index
                and r.get("complete")} == expected]


def summarize(records):
    complete = common_complete(records)
    comparisons, conditions = [], {}
    for history in (False, True):
        for clear in (False, True):
            rows = [r for r in records if r["stage"] == "factorial" and r["opening_index"] in complete
                    and r["history"] == history and r["clear"] == clear]
            conditions[f"history={history},clear={clear}"] = dict(
                games=len(rows), pair_scores=[sum(r["candidate_score"] for r in rows
                                                if r["opening_index"] == i) / 2 for i in complete],
                seconds=sum(r["trace"]["elapsed_seconds"] for r in rows),
                cache_clear_ms=sum(int(line.split("elapsed_ms=")[1]) for r in rows
                                  for search in r["trace"]["searches"] for line in search["info"]
                                  if line.startswith("info string dynamic_weight_cache ")))
    for index in complete:
        for color in (0, 1):
            rows = [r for r in records if r["stage"] == "factorial" and r["opening_index"] == index
                    and r["candidate_color"] == color]
            base = next(r for r in rows if not r["history"] and not r["clear"])
            for row in rows:
                comparisons.append(dict(opening_index=index, color=color, history=row["history"], clear=row["clear"],
                                        same_moves=row["trace"]["moves_usi"] == base["trace"]["moves_usi"],
                                        termination=row["trace"]["termination"],
                                        old_termination=base["trace"]["termination"]))
    return dict(common_complete_openings=complete, conditions=conditions, comparisons=comparisons,
                strength_inference="Insufficient sample; no model/protocol selection or equivalence claim.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    artifacts = run / "artifacts"
    # Every invocation gets an independent directory; interrupted results are never mixed.
    out = artifacts / ("execution-" + time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}")
    out.mkdir(parents=True)
    dump(artifacts / "latest.json", {"directory": str(out)})
    deadline = time.monotonic() + args.budget_seconds
    records, engines = [], []
    status, failure = "starting", None
    planned = 0
    try:
        openings = generated_openings()
        dump(out / "openings.json", openings)
        paths = [args.binary, Path("bin/eval/nn.bin"),
                 Path("YaneuraOu/source/engine/yaneuraou-engine/yaneuraou-search.cpp")]
        configs = []
        for kind, run_name in (("candidate", "weight005"), ("control", "control")):
            release = Path(f"tmp/decision_aligned_{kind}_release").resolve()
            paths.extend(release / name for name in ("backbone.onnx", "head.bin", "head.json"))
            paths.append(Path(f"logs/decision_aligned_{run_name}_from_m0/checkpoints/9.ckpt"))
            configs.append(dict(Threads=1, USI_Hash=16, EvalDir=str(Path("bin/eval").resolve()),
                                ExpertBlendingDir=str(release), ClearTTOnDynamicWeights="true",
                                LogDynamicWeightCache="true", MultiPV=1, USI_Ponder="false", BookFile="no_book"))
        paths.extend(Path(path) for path in (
            "src/train_nnue/run_match.py", "src/train_nnue/match_protocol_pilot.py",
            "tests/test_match_protocol.py",
            "YaneuraOu/extra/onnxruntime/linux/current/lib/libonnxruntime.so.1"))
        paths.extend(run / name for name in ("run.sh", "plan.md", "handoff.md"))
        dump(out / "manifest.json", dict(files={str(p.resolve()): sha256(p) for p in paths},
             options=configs, nodes=100000, max_moves=512, synthetic_source=True,
             pre_opening_history="unknown to players", cshogi_version=getattr(cshogi, "__version__", "unknown"),
             separation="New synthetic walks; no claim of game-level or full training-set independence",
             git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))
        for index, config in enumerate(configs):
            log = f"/tmp/match-protocol-{out.name}-engine{index}.log"
            engine = AuditedEngine(args.binary, config, deadline, log)
            engines.append(engine)
            dump(out / f"engine{index}.json", dict(advertised=engine.advertised, overrides=config, log=log))

        def game(opening, history, clear, color, stage):
            if time.monotonic() >= deadline:
                raise BudgetExpired("50-minute compute deadline")
            for engine in engines:
                engine.setoption("ClearTTOnDynamicWeights", str(clear).lower())
            row = dict(stage=stage, opening_index=opening["index"], history=history, clear=clear,
                       candidate_color=color, complete=False, trace={})
            records.append(row)
            dump(out / "progress.json", dict(stage=stage, opening=opening["index"], history=history,
                                            clear=clear, color=color, completed=sum(r["complete"] for r in records)))
            try:
                first, second = engines if color == 0 else engines[::-1]
                result, _ = play_game(first, second, {"nodes": 100000}, opening["sfen"],
                                      history=history, details=row["trace"])
                if any(search["nodes"] is None for search in row["trace"]["searches"]):
                    raise RuntimeError("missing search nodes")
                row["candidate_score"] = (1 + (result if color == 0 else -result)) / 2
                row["complete"] = True
                row["memory"] = [engine.rss_kib() for engine in engines]
            finally:
                dump(out / f"game-{len(records):03d}.json", row)
                dump(out / "records.json", records)

        # Calibrate on the first opening's history+clear pair. If a factorial
        # sample fits, reuse these two games so the total never exceeds 128.
        for color in (0, 1):
            game(openings[0], True, True, color, "calibration")
        slowest = max(r["trace"]["elapsed_seconds"] for r in records)
        planned = min(16, max(0, int((deadline - time.monotonic() - 30) / (8 * slowest * 1.5))))
        dump(out / "allocation.json", dict(planned_openings=planned, slowest_calibration_seconds=slowest,
                                           safety_factor=1.5, remaining_seconds=deadline-time.monotonic(),
                                           formal_20000_sequential_hours=slowest * 20000 / 3600,
                                           formal_20000_ideal_four_workers_hours=slowest * 20000 / 14400,
                                           estimate_caveat="Synthetic opening estimate; parallel scaling unmeasured"))
        if planned:
            for number, row in enumerate(records, 1):
                row.update(stage="factorial", used_for_timing=True)
                dump(out / f"game-{number:03d}.json", row)
            dump(out / "records.json", records)
        for opening in openings[:planned]:
            for history in (False, True):
                for clear in (False, True):
                    if opening["index"] == 0 and history and clear:
                        continue
                    for color in (0, 1):
                        game(opening, history, clear, color, "factorial")
        status = "complete" if planned else "insufficient_budget"
    except BudgetExpired as error:
        status, failure = "budget_stop", str(error)
    except BaseException as error:
        status, failure = "failed", repr(error)
        raise
    finally:
        for engine in engines:
            try:
                engine.quit()
            except Exception as error:
                status, failure = "failed", f"cleanup: {error!r}; previous: {failure}"
        summary = summarize(records)
        summary.update(status=status, failure=failure, planned_openings=planned,
                       complete_games=sum(r["complete"] for r in records),
                       protocol_gate=("pass" if status == "complete" and planned > 0
                                      and len(summary["common_complete_openings"]) == planned else "hold"))
        dump(out / "summary.json", summary)
        common_count = len(summary["common_complete_openings"])
        short_failure = failure[:2000] if failure else None
        (run / "result-summary.md").write_text(
            f"# H-MATCH-PROTOCOL pilot\n\nStatus: {status}\nFailure: {short_failure}\n"
            f"Protocol gate: {summary['protocol_gate']}\n"
            f"Planned openings: {planned}; common complete openings: {common_count}\n"
            f"\nDetailed artifacts: {out}\n"
            "\nSynthetic legal openings; not an independent strength sample. Timing pair reused only in a complete factorial block.\n"
            "Only openings complete in all 8 cells enter comparisons; partial games remain for diagnosis.\n"
            "No strength/equivalence conclusion. Review regression results, cache events, nodes and terminations.\n"
            "Formal 20,000-game replication requires a separate data-manifest job and compute chunks.\n")


if __name__ == "__main__":
    main()
