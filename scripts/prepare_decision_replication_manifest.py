#!/usr/bin/env python3
"""Historical attempt-001 inventory; not a current admission gate.

Superseded by configs/research_data.json and docs/operations/research-autonomy.md.
Retained with its tests as evidence of the old, overly strict preparation only.
Never use its unconditional held result to block a new experiment.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def inspect_file(root, entry):
    path = root / entry["path"]
    result = dict(entry)
    if not path.is_file():
        return {**result, "exists": False}
    result.update(exists=True, size_bytes=path.stat().st_size)
    # Large historical binaries/logs are deliberately not scanned or hashed.
    if entry.get("mode") == "jsonl_schema":
        if result["size_bytes"] > 32 * 1024 * 1024:
            result["inspection"] = "not_scanned_size_limit"
            return result
        digest = hashlib.sha256()
        count = ids = 0
        fields = set()
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row in {path}")
                count += 1
                fields.update(row)
                ids += bool(row.get("source_namespace") and row.get("source_game_id"))
        result.update(sha256=digest.hexdigest(), rows=count,
                      rows_with_namespaced_game_id=ids, fields=sorted(fields))
    elif entry.get("mode") == "hash_small":
        if result["size_bytes"] > 1024 * 1024:
            raise ValueError(f"unexpected large evidence document: {path}")
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def build_inventory(root, config):
    entries = [inspect_file(root, item) for item in config["inventory"]]
    blockers = list(config["unresolved_requirements"])
    blockers.extend("Missing evidence: " + e["path"] for e in entries if not e["exists"])
    # This task has no supplied new-game corpus or complete ancestry index.
    # Adding a file later must not silently turn an inventory into a proof.
    blockers.append("A new prepare/review must validate the supplied corpus and full exclusion coverage.")
    return {"schema_version": 1, "hypothesis": "H-DECISION-REPLICATION",
            "status": "held", "formal_matches_authorized": False,
            "selected_games": 0, "exclusion_verified": False,
            "scope": "Explicit input inventory only; not an exhaustive dataset search",
            "inventory": entries, "blockers": blockers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    progress = args.output_dir / "progress.json"
    progress.write_text(json.dumps({"phase": "inventory", "complete": False}))
    raw = args.config.read_bytes()
    result = build_inventory(root, json.loads(raw))
    result["config_sha256"] = hashlib.sha256(raw).hexdigest()
    result["runtime"] = {"cshogi": importlib.metadata.version("cshogi"),
                         "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()}
    (args.output_dir / "inventory-manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "summary.md").write_text(
        "# 独立追試manifest: 保留\n\n正式標本0棋譜、対局0局。独立性・除外は未証明。\n\n"
        + "\n".join("- " + item for item in result["blockers"])
        + f"\n\n詳細: {args.output_dir / 'inventory-manifest.json'}\n")
    progress.write_text(json.dumps({"phase": "held", "complete": True,
                                    "formal_matches_authorized": False}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
