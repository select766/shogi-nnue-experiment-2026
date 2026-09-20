"""Offline integration-test stand-in for codex exec; never contacts a service."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

args = sys.argv[1:]
directory = Path(os.environ["RUN_DIR"])
root = Path(args[args.index("--cd") + 1])
output = Path(args[args.index("--output-last-message") + 1])
phase = output.stem
events = root / ".research-loop" / "invocations.jsonl"
with events.open("a") as stream:
    stream.write(json.dumps({"phase": phase, "pid": os.getpid(), "args": args,
                             "prompt": sys.stdin.read()}) + "\n")
mode = (root / ".research-loop" / "mode").read_text()
if mode == "prepare_failure" and phase == "prepare":
    sys.exit(3)
if mode == "malformed" and phase == "prepare":
    output.write_text("not json")
    sys.exit(0)
if phase == "prepare":
    (directory / "plan.md").write_text("Fixed fake test plan\n")
    (directory / "handoff.md").write_text("Offline fixture; result-summary.md is the outcome\n")
    body = f"set -eu\nif kill -0 {os.getpid()} 2>/dev/null; then exit 17; fi\n"
    if mode == "long":
        body += 'sleep 30 &\nprintf "%s" "$!" > "$RUN_DIR/child.pid"\nwait\n'
    elif mode == "compute_failure":
        body += "exit 19\n"
    else:
        body += 'printf "completed offline computation\\n" > "$RUN_DIR/result-summary.md"\n'
    (directory / "run.sh").write_text(body)
    result = {"status": "ready", "summary": "prepared", "timeout_seconds": 1 if mode == "long" else 10}
else:
    if mode == "review_failure":
        sys.exit(5)
    assert (directory / "execution.json").exists()
    (directory / "analysis.md").write_text("Reviewed saved execution, including errors\n")
    report = root / "docs/research/results/fake/README.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("Offline integration result\n")
    subprocess.run(["git", "add", "docs"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "Fake review", "--allow-empty"], cwd=root, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    result = {"status": "complete", "summary": "reviewed", "commit": sha,
              "report": "docs/research/results/fake/README.md", "next_jobs": []}
    if mode == "next_job":
        result["next_jobs"] = [{"id": "followup", "hypothesis": "H-TWO", "task": "Next task",
                                "depends_on": [os.environ["RESEARCH_JOB_ID"]]}]
    if mode == "invalid_review":
        result["next_jobs"] = [{"id": "invalid", "hypothesis": "H-MISSING", "task": "bad", "depends_on": []}]
output.write_text(json.dumps(result))
