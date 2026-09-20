#!/usr/bin/env python3
"""Read-only loop/data readiness check; no model or engine invocation."""
import hashlib
import subprocess

from train_nnue.research_loop import ROOT, Store, choose, read_json, registry


def check_catalog(root):
    catalog = read_json(root / "configs/research_data.json")
    for entry in catalog["datasets"]:
        manifest = root / entry["manifest"]
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != entry["manifest_sha256"]:
            raise ValueError(f"manifest changed: {entry['id']}")
        for name, path in entry["paths"].items():
            if not (root / path).is_file() or not (root / path).stat().st_size:
                raise ValueError(f"missing data: {entry['id']}/{name}: {path}")
        print(f"Available: {entry['id']} (manifest hash verified; files present)")
    return catalog


def main():
    check_catalog(ROOT)
    store = Store(ROOT / ".research-loop")
    state = store.snapshot()
    if store.worker_present() or state["active"]:
        raise ValueError("worker active; readiness check is for a stopped queue")
    if state["paused"] or any(j["status"] == "blocked" for j in state["jobs"]):
        raise ValueError("queue paused or blocked")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise ValueError("uncommitted workspace changes")
    subprocess.run([str(ROOT / "scripts/project_python.sh"), "scripts/check_research_hypotheses.py"],
                   cwd=ROOT, check=True, timeout=300)
    job = choose(state, registry(ROOT))
    print(f"Ready; next job: {job['id'] if job else '(idle; waiting for enqueue/daily)'}")
    print("Does not certify engine/GPU/authentication readiness or rehash large data files.")


if __name__ == "__main__":
    main()
