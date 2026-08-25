#!/usr/bin/env python3
"""Validate the research hypothesis registry and result-document coverage."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "docs/research/hypotheses.md"
RESULTS_DIR = REPO_ROOT / "docs/research/results"

HYPOTHESIS_RE = re.compile(
    r"<!-- hypothesis id=(H-[A-Z0-9-]+) status=([a-z_]+)(?: priority=(\d+))? -->"
)
LINK_RE = re.compile(r"\[[^]]+\]\(([^)]+)\)")
ID_RE = re.compile(r"\bH-[A-Z0-9-]+\b")
OPEN_STATUSES = {"unverified", "in_progress", "held"}
CLOSED_STATUSES = {"supported", "rejected", "saturated", "measured"}


def result_documents() -> set[Path]:
    documents = {path.relative_to(REPO_ROOT) for path in RESULTS_DIR.glob("*/README.md")}
    documents.update(
        path.relative_to(REPO_ROOT)
        for path in RESULTS_DIR.glob("*.md")
        if path.name != "README.md"
    )
    return documents


def validate_registry() -> list[str]:
    errors: list[str] = []
    text = REGISTRY.read_text(encoding="utf-8")
    entries = HYPOTHESIS_RE.findall(text)

    ids = [entry[0] for entry in entries]
    duplicates = sorted({hypothesis_id for hypothesis_id in ids if ids.count(hypothesis_id) > 1})
    if duplicates:
        errors.append(f"duplicate hypothesis IDs: {', '.join(duplicates)}")

    priorities: list[int] = []
    for hypothesis_id, status, priority_text in entries:
        if status not in OPEN_STATUSES | CLOSED_STATUSES:
            errors.append(f"{hypothesis_id}: unknown status {status!r}")
            continue
        if status in OPEN_STATUSES:
            if not priority_text:
                errors.append(f"{hypothesis_id}: open hypothesis has no priority")
            else:
                priorities.append(int(priority_text))
        elif priority_text:
            errors.append(f"{hypothesis_id}: closed hypothesis must not have a priority")

    expected_priorities = list(range(1, len(priorities) + 1))
    if priorities != expected_priorities:
        errors.append(
            "open hypotheses must appear in unique, contiguous priority order from 1: "
            f"found {priorities}, expected {expected_priorities}"
        )

    ledger_match = re.search(
        r"<!-- result-ledger:begin -->(.*?)<!-- result-ledger:end -->",
        text,
        flags=re.DOTALL,
    )
    if ledger_match is None:
        errors.append("result ledger markers are missing")
        return errors

    ledger = ledger_match.group(1)
    registered_results: list[Path] = []
    known_ids = set(ids)
    for line in ledger.splitlines():
        links = LINK_RE.findall(line)
        if not links:
            continue
        if len(links) != 1:
            errors.append(f"ledger row must contain exactly one result link: {line.strip()}")
            continue
        target = (REGISTRY.parent / links[0]).resolve()
        try:
            relative_target = target.relative_to(REPO_ROOT)
        except ValueError:
            errors.append(f"result link escapes repository: {links[0]}")
            continue
        registered_results.append(relative_target)
        row_ids = set(ID_RE.findall(line))
        if not row_ids:
            errors.append(f"ledger row has no hypothesis ID: {links[0]}")
        unknown_ids = sorted(row_ids - known_ids)
        if unknown_ids:
            errors.append(f"{links[0]} references unknown IDs: {', '.join(unknown_ids)}")

    duplicate_results = sorted(
        str(path) for path in set(registered_results) if registered_results.count(path) > 1
    )
    if duplicate_results:
        errors.append(f"result documents registered more than once: {', '.join(duplicate_results)}")

    actual = result_documents()
    registered = set(registered_results)
    missing = sorted(str(path) for path in actual - registered)
    stale = sorted(str(path) for path in registered - actual)
    if missing:
        errors.append(f"unregistered result documents: {', '.join(missing)}")
    if stale:
        errors.append(f"registered result documents do not exist: {', '.join(stale)}")

    return errors


def main() -> int:
    errors = validate_registry()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"OK: {len(HYPOTHESIS_RE.findall(REGISTRY.read_text(encoding='utf-8')))} hypotheses; "
        f"{len(result_documents())} result documents registered"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
