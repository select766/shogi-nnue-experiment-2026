#!/usr/bin/env python3
"""Collect root-grouped positions from actual depth-zero qsearch visits.

The normal searcher samples qsearch entry positions with reservoir sampling.
An independent, fixed evallearn engine then runs exact qsearch on every sampled
entry and moves along its PV to the quiet endpoint used as the NNUE record.
"""

import argparse
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time

import cshogi
import numpy as np


RECORD_BYTES = 40
PSFEN_BYTES = 32
QS_RE = re.compile(r"^qsearch : Value = (-?\d+) , PV =\s*(.*)$")
LEAF_PREFIX = "info string search_leaf "
SUMMARY_PREFIX = "info string search_leaf_summary "


class EngineProcess:
    def __init__(self, command, env=None):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

    def send(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_until(self, predicate):
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(
                    f"engine exited before expected response: {self.process.poll()}"
                )
            line = line.rstrip("\r\n")
            if predicate(line):
                return line

    def initialize(self, options):
        self.send("usi")
        self.read_until(lambda line: line == "usiok")
        for name, value in options:
            self.send(f"setoption name {name} value {value}")
        self.send("isready")
        self.read_until(lambda line: line == "readyok")

    def close(self):
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()


def record_to_sfen(record, board, psfen):
    psfen[0]["sfen"] = np.frombuffer(record[:PSFEN_BYTES], dtype=np.uint8)
    board.set_psfen(psfen)
    # PackedSfen does not encode gamePly. Restore the adjacent PackedSfenValue
    # field so search limits/history and the generated leaf records retain the
    # original game progress.
    game_ply = struct.unpack_from("<H", record, 36)[0]
    position = board.sfen().rsplit(" ", 1)[0]
    return f"{position} {max(1, game_ply)}"


def endpoint_record(sfen, score, pv):
    board = cshogi.Board(sfen)
    entry_turn = board.turn
    for move in pv:
        if move in {"", "resign", "win", "none"}:
            continue
        board.push_usi(move)
    if board.turn != entry_turn:
        score = -score
    score = max(-32000, min(32000, score))
    packed = np.zeros(1, dtype=cshogi.PackedSfen)
    board.to_psfen(packed)
    record = bytearray(RECORD_BYTES)
    record[:PSFEN_BYTES] = packed[0]["sfen"].tobytes()
    struct.pack_into(
        "<hHHbB", record, 32, score, 0, min(board.move_number, 65535), 0, 0
    )
    return bytes(record)


def collect_entries(searcher, root_sfen, nodes, expected):
    searcher.send(f"position sfen {root_sfen}")
    searcher.send(f"go nodes {nodes}")
    entries = []
    visits = None
    while True:
        line = searcher.read_until(lambda _: True)
        if line.startswith(SUMMARY_PREFIX):
            fields = line[len(SUMMARY_PREFIX) :].split()
            visits = int(fields[1])
        elif line.startswith(LEAF_PREFIX):
            fields = line[len(LEAF_PREFIX) :].split(maxsplit=1)
            if len(fields) == 2:
                entries.append(fields[1])
        elif line.startswith("bestmove "):
            break
    if len(entries) != expected:
        return None, visits
    return entries, visits


def exact_qsearch(labeler, sfen):
    labeler.send(f"position sfen {sfen}")
    labeler.send("qsearch")
    line = labeler.read_until(lambda value: value.startswith("qsearch : "))
    match = QS_RE.match(line)
    if not match:
        raise ValueError(f"unexpected qsearch response: {line}")
    score = int(match.group(1))
    pv = match.group(2).split()
    return endpoint_record(sfen, score, pv)


def write_metadata(
    path, args, completed, skipped, source_consumed, visit_sum, elapsed, status
):
    metadata = {
        "format": "root-grouped-paired-v1",
        "num_roots": completed,
        "group_size": args.group_size,
        "record_bytes": RECORD_BYTES,
        "leaf_distribution": "algorithm-r-reservoir-over-depth-zero-qsearch-visits",
        "search_threads": 1,
        "search_nodes": args.nodes,
        "search_seed": args.seed,
        "search_engine": str(args.search_engine.resolve()),
        "search_eval_dir": str(args.search_eval_dir.resolve()),
        "search_expert_blending_dir": (
            str(args.search_expert_blending_dir.resolve())
            if args.search_expert_blending_dir else None
        ),
        "label_engine": str(args.label_engine.resolve()),
        "label_eval_dir": str(args.label_eval_dir.resolve()),
        "label": "exact qsearch score at sampled entry, sign-adjusted at qsearch PV endpoint",
        "source_roots": str(args.roots_file.resolve()),
        "source_start_root": args.start_root,
        "source_records_consumed": source_consumed,
        "skipped_roots": skipped,
        "mean_qsearch_boundary_visits": visit_sum / completed if completed else None,
        "elapsed_seconds": elapsed,
        "status": status,
    }
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-engine", type=Path, required=True)
    parser.add_argument("--search-eval-dir", type=Path, required=True)
    parser.add_argument(
        "--search-expert-blending-dir",
        type=Path,
        help="Enable the exported router/experts while collecting search leaves",
    )
    parser.add_argument("--label-engine", type=Path, required=True)
    parser.add_argument("--label-eval-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--max-roots", type=int, required=True)
    parser.add_argument("--start-root", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.group_size < 2 or args.group_size > 64:
        parser.error("group-size must be in [2, 64]")
    if args.nodes <= 0 or args.max_roots <= 0 or args.start_root < 0:
        parser.error("nodes/max-roots must be positive and start-root non-negative")
    roots_size = args.roots_file.stat().st_size
    if roots_size % RECORD_BYTES:
        raise ValueError("roots file size is not a multiple of 40 bytes")
    available = roots_size // RECORD_BYTES
    if args.start_root >= available:
        raise ValueError("start-root is outside the source roots file")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots_output = args.output_dir / "roots.bin"
    leaves_output = args.output_dir / "leaves.bin"
    metadata_path = args.output_dir / "metadata.json"
    completed = 0
    skipped = 0
    source_consumed = 0
    visit_sum = 0
    if args.resume and roots_output.exists() and leaves_output.exists():
        if roots_output.stat().st_size % RECORD_BYTES:
            raise ValueError("partial roots.bin is misaligned")
        completed = roots_output.stat().st_size // RECORD_BYTES
        expected_leaves = completed * args.group_size * RECORD_BYTES
        if leaves_output.stat().st_size != expected_leaves:
            raise ValueError("partial leaves.bin does not match roots.bin")
        if not metadata_path.exists():
            raise ValueError("resume requires metadata.json")
        previous = json.loads(metadata_path.read_text())
        source_consumed = int(previous["source_records_consumed"])
        skipped = int(previous["skipped_roots"])
        mean_visits = previous.get("mean_qsearch_boundary_visits")
        visit_sum = int(round(float(mean_visits) * completed)) if mean_visits else 0
    elif roots_output.exists() or leaves_output.exists():
        raise FileExistsError("output exists; pass --resume or choose another directory")

    env = os.environ.copy()
    search_lib = (
        args.search_engine.resolve().parent.parent
        / "extra/onnxruntime/linux/current/lib"
    )
    env["LD_LIBRARY_PATH"] = str(search_lib) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    searcher = EngineProcess([str(args.search_engine.resolve())], env=env)
    labeler = EngineProcess([str(args.label_engine.resolve())])
    search_options = [
            ("Threads", 1),
            ("USI_Hash", args.hash_mb),
            ("EvalDir", args.search_eval_dir.resolve()),
            ("USI_OwnBook", "false"),
            ("SearchLeafSamples", args.group_size),
            ("SearchLeafSeed", args.seed),
    ]
    if args.search_expert_blending_dir:
        search_options.append(
            ("ExpertBlendingDir", args.search_expert_blending_dir.resolve())
        )
    searcher.initialize(search_options)
    labeler.initialize(
        [
            ("Threads", 1),
            ("USI_Hash", args.hash_mb),
            ("EvalDir", args.label_eval_dir.resolve()),
            ("USI_OwnBook", "false"),
        ]
    )

    mode = "ab" if completed else "wb"
    started = time.monotonic()
    board = cshogi.Board()
    psfen = np.zeros(1, dtype=cshogi.PackedSfen)
    try:
        with args.roots_file.open("rb") as source, roots_output.open(mode) as root_file, \
                leaves_output.open(mode) as leaf_file:
            source.seek((args.start_root + source_consumed) * RECORD_BYTES)
            while completed < args.max_roots and args.start_root + source_consumed < available:
                root_record = source.read(RECORD_BYTES)
                if len(root_record) != RECORD_BYTES:
                    break
                source_consumed += 1
                root_sfen = record_to_sfen(root_record, board, psfen)
                entries, visits = collect_entries(
                    searcher, root_sfen, args.nodes, args.group_size
                )
                if entries is None:
                    skipped += 1
                    continue
                leaf_records = [exact_qsearch(labeler, entry) for entry in entries]
                root_file.write(root_record)
                leaf_file.write(b"".join(leaf_records))
                completed += 1
                visit_sum += visits or 0
                if completed % args.progress_every == 0:
                    root_file.flush()
                    leaf_file.flush()
                    elapsed = time.monotonic() - started
                    write_metadata(
                        metadata_path, args, completed, skipped, source_consumed, visit_sum,
                        elapsed, "in_progress"
                    )
                    rate = completed / elapsed
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "skipped": skipped,
                                "roots_per_second": rate,
                                "eta_seconds": (args.max_roots - completed) / rate,
                            }
                        ),
                        flush=True,
                    )
            root_file.flush()
            os.fsync(root_file.fileno())
            leaf_file.flush()
            os.fsync(leaf_file.fileno())
    finally:
        searcher.close()
        labeler.close()
    elapsed = time.monotonic() - started
    status = "complete" if completed == args.max_roots else "source_exhausted"
    write_metadata(
        metadata_path, args, completed, skipped, source_consumed, visit_sum, elapsed, status
    )
    print(metadata_path.read_text(), end="")


if __name__ == "__main__":
    main()
