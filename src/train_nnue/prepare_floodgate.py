"""Provenance-preserving rated CSA corpus and fixed, game-disjoint evaluation pools.

No engine searches. Raw archive is retained; every streamed entry is CRC checked.
Never treats missing ratings/scores as zero or reported moves as perfect labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import zlib

import cshogi

ROOT = Path(__file__).resolve().parents[2]
MOVE = re.compile(r"^[+-][0-9]{4}(?:FU|KY|KE|GI|KI|KA|HI|OU|TO|NY|NK|NG|UM|RY)$")
ENDS = {"%TORYO", "%SENNICHITE", "%JISHOGI", "%KACHI", "%TSUMI", "%MAX_MOVES", "%OUTE_SENNICHITE"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashed(value):
    return hashlib.sha256(value.encode()).hexdigest()


def identity(sfen):
    return " ".join(sfen.split()[:3])


def dump(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def archive_entries(archive):
    listing = subprocess.check_output(["7z", "l", "-slt", str(archive)], text=True, timeout=60)
    entries = []
    for block in listing.split("----------\n", 1)[1].strip().split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        name = fields["Path"]
        path = PurePosixPath(name)
        mode = fields["Attributes"].split()[-1]
        if path.is_absolute() or ".." in path.parts or mode[0] not in "d-":
            raise ValueError(f"unsafe archive member: {name}")
        if mode.startswith("d"):
            continue
        size = int(fields["Size"])
        if not name.endswith(".csa") or size > 4 * 1024**2 or fields["Encrypted"] != "-":
            raise ValueError(f"unexpected archive member: {name}")
        entries.append((name, size, int(fields["CRC"], 16)))
    return entries


def stream_archive(archive, entries):
    # 7z emits the archive order. Per-entry CRC detects any ordering discrepancy.
    process = subprocess.Popen(["7z", "x", "-so", "-bd", "-bb0", str(archive)], stdout=subprocess.PIPE)
    try:
        for name, size, crc in entries:
            raw = process.stdout.read(size)
            if len(raw) != size or zlib.crc32(raw) != crc:
                raise ValueError(f"archive CRC/length/order mismatch: {name}")
            yield name, raw
        if process.stdout.read(1) or process.wait() != 0:
            raise ValueError("archive extraction failed or has trailing data")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class Rejected(ValueError):
    pass


def parse_game(name, raw, config):
    try:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("cp932")
        encoding = "cp932"
    lines = text.splitlines()
    names, ratings, rate_sources = [], [], []
    for sign, side in (("+", "black"), ("-", "white")):
        players = [line[2:] for line in lines if line.startswith("N" + sign)]
        rate_lines = [line for line in lines if line.startswith(f"'{side}_rate:")]
        if len(players) != 1 or len(rate_lines) != 1:
            raise Rejected("missing_or_duplicate_rating_or_name")
        player = players[0]
        try:
            account, number = rate_lines[0][12:].rsplit(":", 1)
            rating = float(number)
        except ValueError:
            raise Rejected("invalid_rating") from None
        if not math.isfinite(rating) or account.rsplit("+", 1)[0] != player:
            raise Rejected("invalid_rating_or_player_mismatch")
        if rating < config["min_rating"]:
            raise Rejected("rating_below_threshold")
        names.append(player)
        ratings.append(rating)
        rate_sources.append(rate_lines[0])
    info = {}
    for line in lines:
        if line.startswith("$") and ":" in line:
            key, value = line[1:].split(":", 1)
            if key in info:
                raise Rejected("duplicate_header")
            info[key] = value
    if info.get("EVENT") != PurePosixPath(name).stem:
        raise Rejected("event_filename_mismatch")
    try:
        started = datetime.strptime(info["START_TIME"], "%Y/%m/%d %H:%M:%S")
    except (KeyError, ValueError):
        raise Rejected("missing_or_invalid_start_time") from None
    if started.year != config["year"]:
        raise Rejected("wrong_year")
    # Restrict research pools to replayable, complete standard-start games.
    # All rejected originals remain in the immutable archive.
    position = [line for line in lines if line.startswith("P") or line in ("+", "-")]
    standard = cshogi.Board().csa_pos().strip().splitlines()
    if position != standard and position != ["PI", "+"]:
        raise Rejected("nonstandard_start")
    summaries = [line[len("'summary:"):] for line in lines if line.startswith("'summary:")]
    if len(summaries) > 1:
        raise Rejected("duplicate_summary")
    summary = summaries[0] if summaries else None
    reason = summary.split(":", 1)[0] if summary else None
    if reason is not None and reason not in {"toryo", "kachi", "sennichite", "oute_sennichite", "max_moves", "jishogi", "tsumi"}:
        raise Rejected("abnormal_summary")
    endings = set(line.split(",", 1)[0] for line in lines if line.startswith("%"))
    if not endings and reason in {"max_moves", "oute_sennichite", "sennichite", "jishogi"}:
        endings = {"%" + reason.upper()}
    if len(endings) != 1 or not endings <= ENDS:
        raise Rejected("incomplete_or_abnormal_end")
    board = cshogi.Board()
    moves, positions, scores, times = [], [], [], []
    ended = False
    for line in lines:
        if line.startswith("%"):
            ended = True
        elif line.startswith(("+", "-")) and len(line) > 1:
            if ended or not MOVE.fullmatch(line) or (line[0] == "-") != bool(board.turn):
                raise Rejected("invalid_move_record")
            move = board.move_from_csa(line[1:])
            if not move or not board.is_legal(move):
                raise Rejected("illegal_move")
            positions.append(board.sfen())
            moves.append(cshogi.move_to_usi(move))
            scores.append(None)
            times.append(None)
            board.push(move)
        elif not ended and line.startswith("'** "):
            if not moves:
                raise Rejected("score_before_move")
            token = line[4:].split()[0] if line[4:].split() else ""
            scores[-1] = int(token) if re.fullmatch(r"[+-]?\d+", token) else None
        elif not ended and line.startswith("T"):
            if not moves or not re.fullmatch(r"T\d+", line):
                raise Rejected("invalid_time")
            times[-1] = int(line[1:])
    if not moves:
        raise Rejected("empty_game")
    end = next(iter(endings))
    winner = 0 if end in {"%SENNICHITE", "%JISHOGI", "%MAX_MOVES"} else (
        1 + board.turn if end == "%KACHI" else 2 - board.turn)
    if end == "%MAX_MOVES":
        limits = [line.split(":", 1)[1] for line in lines if line.startswith("'Max_Moves:")]
        if len(limits) != 1 or not limits[0].isdigit() or len(moves) != int(limits[0]):
            raise Rejected("max_moves_length_mismatch")
    if summary:
        outcomes = summary.split(":")[1:]
        possibilities = {1: {names[0] + " win", names[1] + " lose"},
                         2: {names[0] + " lose", names[1] + " win"},
                         0: {names[0] + " draw", names[1] + " draw"}}
        matches = [value for value, expected in possibilities.items() if set(outcomes) == expected and len(outcomes) == 2]
        if len(matches) != 1:
            raise Rejected("invalid_result_summary")
        if end != "%OUTE_SENNICHITE" and matches[0] != winner:
            raise Rejected("result_summary_mismatch")
        winner = matches[0]
    game_hash = hashed(cshogi.STARTING_SFEN + "\n" + " ".join(moves))
    game = {"source_namespace": "floodgate2025", "source_game_id": info["EVENT"],
            "archive_member": name, "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "encoding": encoding, "started_at": started.isoformat() + "+09:00",
            "players": names, "ratings": ratings, "rating_lines": rate_sources,
            "initial_sfen": cshogi.STARTING_SFEN, "moves": moves, "total_plies": len(moves),
            "game_hash": game_hash, "endgame": end, "gameResult": winner,
            "server_summary": summary,
            "reported_scores_black": scores, "move_seconds": times,
            "score_source": "CSA '**' player-reported; not fixed-node teacher"}
    return game, positions


def split_for(game_hash, seed):
    bucket = int(hashed(seed + "|" + game_hash)[:8], 16) % 100
    return "development" if bucket < 30 else "validation" if bucket < 50 else "test" if bucket < 70 else "match"


def load_exclusions(config):
    excluded, sources = set(), []
    for key in ("exclude_jsonl", "exclude_match_json"):
        for name in config[key]:
            path = ROOT / name
            with path.open() as stream:
                rows = (json.loads(line) for line in stream if line.strip()) if key == "exclude_jsonl" else json.load(stream)["details"]
                sfens = {identity(r["sfen"]) for r in rows}
            excluded.update(sfens)
            sources.append({"path": name, "sha256": sha(path), "unique_positions": len(sfens)})
    return excluded, sources


def position_record(game, sfen, index):
    turn = index % 2
    score = game["reported_scores_black"][index]
    record = {"source_namespace": game["source_namespace"], "source_game_id": game["source_game_id"],
              "game_hash": game["game_hash"], "sfen": sfen, "turn": turn,
              "game_ply": index + 1, "played_plies": index, "bestmove": game["moves"][index],
              "label_source": "recorded_game_move", "gameResult": game["gameResult"],
              "move_seconds": game["move_seconds"][index], "reported_score_black": score,
              "eval_source": "player_reported_CSA_black_to_side_to_move"}
    if score is not None:
        record["eval"] = score if turn == 0 else -score
    return record


def build(config, output, *, limit=None):
    archive = ROOT / config["archive"]
    if sha(archive) != config["archive_sha256"]:
        raise ValueError("archive SHA256 differs from official pinned checksum")
    if config["min_rating"] < 3500:
        raise ValueError("both player ratings must be at least 3500")
    entries = archive_entries(archive)
    excluded, sources = load_exclusions(config)
    output.mkdir(parents=True, exist_ok=False)
    stats = Counter()
    months, players, phases, splits = Counter(), Counter(), Counter(), Counter()
    seen_games, used_samples = set(), set(excluded)
    files = []
    with ExitStack() as stack:
        def writer(name, compressed=False):
            files.append(name)
            if compressed:
                raw = stack.enter_context((output / name).open("wb"))
                compressed_stream = gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0)
                return stack.enter_context(io.TextIOWrapper(compressed_stream, encoding="utf-8"))
            return stack.enter_context((output / name).open("w"))
        games = writer("games.jsonl.gz", True)
        positions_out = writer("positions.jsonl.gz", True)
        rejected = writer("rejections.jsonl.gz", True)
        samples = {split: writer(split + ".jsonl") for split in ("development", "validation", "test", "match")}
        members = stream_archive(archive, entries)
        try:
            for member, raw in members:
                stats["archive_games_scanned"] += 1
                try:
                    game, sfens = parse_game(member, raw, config)
                except (Rejected, UnicodeError, ValueError) as error:
                    reason = str(error) if isinstance(error, Rejected) else type(error).__name__
                    stats["rejected_" + reason] += 1
                    dump(rejected, {"archive_member": member, "reason": reason, "raw_sha256": hashlib.sha256(raw).hexdigest()})
                else:
                    if game["game_hash"] in seen_games:
                        stats["rejected_duplicate_game"] += 1
                        dump(rejected, {"archive_member": member, "reason": "duplicate_game", "game_hash": game["game_hash"]})
                    else:
                        seen_games.add(game["game_hash"])
                        split = split_for(game["game_hash"], config["seed"])
                        game["split"] = split
                        dump(games, game)
                        stats["accepted_games"] += 1
                        months[game["started_at"][:7]] += 1
                        players.update(game["players"])
                        splits[split] += 1
                        candidates = []
                        for i, sfen in enumerate(sfens):
                            row = position_record(game, sfen, i)
                            row["split"] = split
                            row["overlaps_known_old_position"] = identity(sfen) in excluded
                            dump(positions_out, row)
                            stats["positions"] += 1
                            if row["overlaps_known_old_position"]:
                                stats["positions_overlapping_known_old"] += 1
                            band = "1-40" if i < 40 else "41-100" if i < 100 else "101+"
                            phases[band] += 1
                            if i + 1 < config["sample_min_ply"] or identity(sfen) in used_samples:
                                continue
                            # Skip instantaneous/book moves in fixed samples; retained in full corpus.
                            if row["move_seconds"] is None or row["move_seconds"] == 0:
                                continue
                            if split == "match" and ("eval" not in row or abs(row["eval"]) > config["opening_max_abs_score"]):
                                continue
                            candidates.append((hashed(config["seed"] + "|" + game["game_hash"] + "|" + str(i)), row))
                        if candidates:
                            row = min(candidates, key=lambda v: v[0])[1]
                            row["initial_sfen"] = game["initial_sfen"]
                            row["history_usi"] = game["moves"][:row["played_plies"]]
                            row["ratings"] = game["ratings"]
                            used_samples.add(identity(row["sfen"]))
                            dump(samples[split], row)
                            stats["samples_" + split] += 1
                            ply = row["game_ply"]
                            stats["samples_ply_" + ("1-40" if ply <= 40 else "41-100" if ply <= 100 else "101+")] += 1
                        else:
                            stats["games_without_sample_" + split] += 1
                if stats["archive_games_scanned"] % 5000 == 0:
                    save(output / "progress.json", dict(stats))
                    print(dict(stats), flush=True)
                if limit is not None and stats["archive_games_scanned"] >= limit:
                    break
        finally:
            members.close()
    complete = stats["archive_games_scanned"] == len(entries) and limit is None
    manifest = {"schema_version": 1, "complete": complete, "created_at": datetime.now(timezone.utc).isoformat(),
                "config": config, "archive_members": len(entries), "statistics": dict(stats),
                "games_by_month": dict(sorted(months.items())), "games_by_split": dict(splits),
                "positions_by_ply_band": dict(phases), "player_game_counts": dict(players.most_common()),
                "exclusions": sources, "excluded_unique_positions": len(excluded),
                "runtime": {"cshogi": importlib.metadata.version("cshogi")},
                "code_sha256": sha(Path(__file__)),
                "files": {name: {"sha256": sha(output / name), "bytes": (output / name).stat().st_size} for name in files},
                "limitations": ["Original game moves, not 1M-node teacher labels",
                    "Reported scores are heterogeneous player annotations; missing scores are absent, not zero",
                    "No claim of complete disjointness from all pretrained ancestors",
                    "Full positions retain transpositions/repetitions; fixed samples are unique and one per game",
                    "Game-disjoint pools may share openings; game identity is full initial position + move sequence"]}
    save(output / "manifest.json", manifest)
    save(output / "progress.json", {"complete": complete, **dict(stats)})
    print(json.dumps({"output": str(output), "complete": complete, "statistics": dict(stats)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/floodgate2025.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="Smoke test only: manifest complete=false")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    config = json.loads(args.config.read_text())
    build(config, args.output or ROOT / config["output"], limit=args.limit)


if __name__ == "__main__":
    main()
