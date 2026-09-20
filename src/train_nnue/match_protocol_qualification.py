"""Real USI forced-cycle fixtures; no unrestricted matches or strength inference."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import time

from train_nnue.match_protocol_pilot import AuditedEngine, dump, sha256
from train_nnue.run_match import play_game


FIXTURES = (
    ("repetition_draw", "4k4/9/9/9/9/9/9/9/4K4 b - 1",
     ("5i6i", "5a6a", "6i5i", "6a5a"), 0),
    ("perpetual_check_black", "4k4/9/4R4/9/9/9/9/9/K8 w - 1",
     ("5a4a", "5c4c", "4a5a", "4c5c"), -1),
    ("perpetual_check_white", "k8/9/9/9/9/9/4r4/9/4K4 b - 1",
     ("5i4i", "5g4g", "4i5i", "4g5g"), 1),
)


class ForcedCycle:
    """Only accept an actual engine bestmove from the single allowed move."""
    def __init__(self, engine, cycle):
        self.engine, self.moves = engine, iter(cycle * 3)

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def go(self, *, nodes, listener):
        expected = next(self.moves)
        result = self.engine.go(nodes=nodes, listener=listener, searchmoves=[expected])
        if result[0] != expected:
            raise RuntimeError(f"forced move mismatch: {expected} != {result[0]}")
        return result


def runtime_manifest(paths):
    return dict(
        python=platform.python_version(), platform=platform.platform(),
        distributions={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        cshogi_version=importlib.metadata.version("cshogi"),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        submodules=subprocess.check_output(["git", "submodule", "status"], text=True),
        files={str(Path(p).resolve()): sha256(p) for p in paths})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--budget-seconds", type=int, default=600)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "artifacts" / (time.strftime("execution-%Y%m%dT%H%M%S") + f"-{os.getpid()}")
    out.mkdir(parents=True)
    dump(run / "artifacts/latest.json", dict(directory=str(out)))
    deadline = time.monotonic() + args.budget_seconds
    records, engine = [], None
    status, failure = "failed", "interrupted before completion"
    try:
        paths = [args.binary, "uv.lock", "pyproject.toml", "bin/eval/nn.bin",
                 "src/train_nnue/run_match.py", "src/train_nnue/match_protocol_pilot.py",
                 "src/train_nnue/match_protocol_qualification.py",
                 "tests/test_match_protocol_qualification.py",
                 "YaneuraOu/extra/onnxruntime/linux/current/lib/libonnxruntime.so.1"]
        paths += [run / name for name in ("run.sh", "plan.md", "handoff.md", "data-catalog.json", "input-manifest.json")]
        for model in ("candidate", "control"):
            paths += [Path(f"tmp/decision_aligned_{model}_release") / name
                      for name in ("backbone.onnx", "head.bin", "head.json")]
        manifest = runtime_manifest(paths)
        dump(out / "manifest.json", manifest)
        # Refuse changed model/input files relative to prepare.
        import json
        expected = json.loads((run / "input-manifest.json").read_text())
        for path, digest in expected.items():
            if sha256(path) != digest:
                raise RuntimeError(f"input changed since prepare: {path}")
        for model in ("candidate", "control"):
            options = dict(Threads=1, USI_Hash=16, MultiPV=1, USI_Ponder="false",
                           BookFile="no_book", ResignValue=99999,
                           EvalDir=str(Path("bin/eval").resolve()),
                           ExpertBlendingDir=str(Path(f"tmp/decision_aligned_{model}_release").resolve()),
                           ClearTTOnDynamicWeights="true", LogDynamicWeightCache="true")
            log = f"/tmp/match-qualification-{out.name}-{model}.log"
            engine = AuditedEngine(args.binary, options, deadline, log)
            dump(out / f"engine-{model}.json", dict(options=options, advertised=engine.advertised, log=log))
            # Store the actual loaded native libraries, not just an assumed path.
            maps = Path(f"/proc/{engine.process.pid}/maps").read_text()
            (out / f"maps-{model}.txt").write_text(maps)
            libraries = sorted({line.split()[-1] for line in maps.splitlines()
                                if line.split()[-1].startswith("/") and ".so" in line.split()[-1]})
            dump(out / f"libraries-{model}.json", {p: sha256(p) for p in libraries})
            for name, sfen, cycle, result in FIXTURES:
                for history in (False, True):
                    for clear in (False, True):
                        engine.setoption("ClearTTOnDynamicWeights", str(clear).lower())
                        for limit in (8, 12):
                            row = dict(model=model, fixture=name, history=history, clear=clear,
                                       limit=limit, complete=False, trace={})
                            records.append(row)
                            dump(out / "progress.json", {k: v for k, v in row.items() if k != "trace"})
                            try:
                                forced = ForcedCycle(engine, cycle)
                                actual = play_game(forced, forced, {"nodes": 128}, sfen,
                                                   max_moves=limit, history=history, details=row["trace"])
                                expected_result = (0, 8) if limit == 8 else (result, 12)
                                expected_reason = "max_moves" if limit == 8 else name
                                if actual != expected_result or row["trace"]["termination"] != expected_reason:
                                    raise RuntimeError(f"fixture failed: {actual}, {row['trace']['termination']}")
                                if len(row["trace"]["searches"]) != limit:
                                    raise RuntimeError("unexpected search count")
                                row["complete"] = True
                            finally:
                                dump(out / f"fixture-{len(records):03d}.json", row)
                                dump(out / "records.json", records)
            engine.quit()
            engine = None
        status, failure = "pass", None
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception as error:
                status, failure = "failed", f"cleanup: {error!r}; previous: {failure}"
        dump(out / "summary.json", dict(status=status, failure=failure, expected=48,
             complete=sum(r["complete"] for r in records), strength_inference="not measured"))
        (run / "result-summary.md").write_text(
            f"# H-MATCH-PROTOCOL qualification\n\nStatus: {status}\nFailure: {str(failure)[:2000]}\n"
            f"Completed fixtures: {sum(r['complete'] for r in records)}/48\nArtifacts: {out}\n\n"
            "Single-searchmove real-engine fixtures verify USI integration and runner adjudication.\n"
            "They do not prove autonomous repetition choice or engine-internal repetition scores.\n"
            "No strength matches performed; preregistered sensitivity experiment remains a separate job.\n")
    if status != "pass":
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
