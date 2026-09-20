import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

import cshogi

from train_nnue.prepare_floodgate import (Rejected, archive_entries, identity,
    build, parse_game, position_record, sha, split_for, stream_archive)
from scripts.verify_floodgate2025 import verify
from scripts.select_floodgate_openings import select

NAME = "2025/wdoor+floodgate-300-10F+Alpha+Beta+20250101000000.csa"
CONFIG = {"min_rating": 3500, "year": 2025}


def fixture(black="3500", white="3500"):
    return ("V2\nN+Alpha\nN-Beta\n$EVENT:" + Path(NAME).stem +
            "\n$START_TIME:2025/01/01 00:00:00\nPI\n+\n"
            f"'black_rate:Alpha+hash:{black}\n'white_rate:Beta+hash:{white}\n"
            "+7776FU\nT2\n'** 15\n-3334FU\nT3\n'** -25\n+2726FU\nT1\n%TORYO\n").encode()


class FloodgateTest(unittest.TestCase):
    def test_both_ratings_inclusive_and_unknown_rejected(self):
        game, _ = parse_game(NAME, fixture(), CONFIG)
        self.assertEqual(game["ratings"], [3500, 3500])
        for black, white in (("3499.9", "4200"), ("4200", "3499.9"), ("nan", "4000"), ("inf", "4000"), ("?", "4000")):
            with self.subTest(black=black, white=white), self.assertRaises(Rejected):
                parse_game(NAME, fixture(black, white), CONFIG)
        with self.assertRaises(Rejected):
            parse_game(NAME, fixture().replace(b"'white_rate:", b"'unknown_rate:"), CONFIG)

    def test_ratings_must_belong_to_players(self):
        with self.assertRaisesRegex(Rejected, "mismatch"):
            parse_game(NAME, fixture().replace(b"Alpha+hash", b"Different+hash"), CONFIG)

    def test_real_ply_history_scores_and_winner(self):
        game, sfens = parse_game(NAME, fixture(), CONFIG)
        self.assertEqual(game["moves"], ["7g7f", "3c3d", "2g2f"])
        self.assertEqual(game["gameResult"], 1)
        board = cshogi.Board(game["initial_sfen"])
        for i, move in enumerate(game["moves"]):
            self.assertEqual(board.sfen(), sfens[i])
            row = position_record(game, sfens[i], i)
            self.assertEqual(row["game_ply"], i + 1)
            self.assertEqual(row["played_plies"], i)
            self.assertEqual(row["bestmove"], move)
            board.push_usi(move)
        self.assertEqual(position_record(game, sfens[1], 1)["eval"], 25)
        self.assertNotIn("eval", position_record(game, sfens[2], 2))

    def test_invalid_moves_and_incomplete_games_rejected(self):
        for raw in (fixture().replace(b"+7776FU", b"+7775FU"), fixture().replace(b"-3334FU", b"+3334FU"),
                    fixture().replace(b"%TORYO", b"%CHUDAN"), fixture().replace(b"%TORYO", b""),
                    fixture().replace(b"PI\n+", b"PI82HI\n+")):
            with self.subTest(raw=raw), self.assertRaises(Rejected):
                parse_game(NAME, raw, CONFIG)

    def test_year_and_event_validated(self):
        for raw in (fixture().replace(b"2025/01/01", b"2024/01/01"),
                    fixture().replace(b"$EVENT:wdoor", b"$EVENT:other")):
            with self.assertRaises(Rejected):
                parse_game(NAME, raw, CONFIG)

    def test_duplicate_declaration_and_server_only_endings(self):
        raw = fixture().replace(b"%TORYO", b"%KACHI\n%KACHI") + b"'summary:kachi:Alpha lose:Beta win\n"
        game, _ = parse_game(NAME, raw, CONFIG)
        self.assertEqual(game["endgame"], "%KACHI")
        self.assertEqual(game["gameResult"], 2)
        raw = fixture().replace(b"%TORYO", b"'Max_Moves:3\n'summary:max_moves:Alpha draw:Beta draw")
        game, _ = parse_game(NAME, raw, CONFIG)
        self.assertEqual(game["endgame"], "%MAX_MOVES")
        self.assertEqual(game["gameResult"], 0)
        with self.assertRaisesRegex(Rejected, "length_mismatch"):
            parse_game(NAME, raw.replace(b"Max_Moves:3", b"Max_Moves:512"), CONFIG)
        with self.assertRaises(Rejected):
            parse_game(NAME, fixture() + b"'summary:illegal kachi:Alpha lose:Beta win\n", CONFIG)
        with self.assertRaises(Rejected):
            parse_game(NAME, fixture() + b"'summary:toryo:Alpha lose:Beta win\n", CONFIG)

    def test_content_dedup_and_split_ignore_comments(self):
        game, _ = parse_game(NAME, fixture(), CONFIG)
        other, _ = parse_game(NAME, fixture() + b"'another comment\n", CONFIG)
        self.assertEqual(game["game_hash"], other["game_hash"])
        self.assertNotEqual(game["raw_sha256"], other["raw_sha256"])
        self.assertEqual(split_for(game["game_hash"], "seed"), split_for(other["game_hash"], "seed"))
        self.assertEqual(identity("a b c 1"), identity("a b c 99"))

    def test_archive_paths_checked(self):
        for name, mode in (("../escape.csa", "-rw-------"), ("/escape.csa", "-rw-------"), ("x.csa", "lrwxrwxrwx")):
            listing = f"header\n----------\nPath = {name}\nSize = 1\nAttributes = A_ {mode}\nCRC = 00000000\nEncrypted = -\n"
            with patch("subprocess.check_output", return_value=listing), self.assertRaises(ValueError):
                archive_entries(Path("unused"))

    def test_real_archive_stream_and_crc(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.csa").write_bytes(fixture())
            (root / "b.csa").write_bytes(fixture("4200", "4200"))
            archive = root / "test.7z"
            subprocess.run(["7z", "a", str(archive), "a.csa", "b.csa"], cwd=root,
                           check=True, stdout=subprocess.DEVNULL)
            entries = archive_entries(archive)
            self.assertEqual(dict(stream_archive(archive, entries)), {p.name: p.read_bytes() for p in root.glob("*.csa")})
            wrong = [(n, s, c ^ 1) for n, s, c in entries]
            with self.assertRaisesRegex(ValueError, "CRC"):
                list(stream_archive(archive, wrong))

    def test_complete_build_is_replayable_deduplicated_and_repeatable(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in range(20):
                name = f"game-{number}.csa"
                header = fixture().decode().split("+7776FU")[0].replace(Path(NAME).stem, Path(name).stem)
                board = cshogi.Board()
                for ply in range(3):
                    move = list(board.legal_moves)[number if ply == 0 else 0]
                    header += ("+" if board.turn == 0 else "-") + cshogi.move_to_csa(move) + "\nT1\n'** 20\n"
                    board.push(move)
                (root / name).write_text(header + "%TORYO\n")
            archive = root / "test.7z"
            subprocess.run(["7z", "a", str(archive), "*.csa"], cwd=root, check=True, stdout=subprocess.DEVNULL)
            config = {**CONFIG, "archive": str(archive), "archive_sha256": sha(archive),
                      "exclude_jsonl": [], "exclude_match_json": [], "sample_min_ply": 1,
                      "opening_max_abs_score": 500, "seed": "test"}
            first, second = root / "first", root / "second"
            build(config, first)
            build(config, second)
            self.assertEqual(verify(first)["games"], 20)
            self.assertEqual(verify(first)["positions"], 60)
            self.assertEqual(json.loads((first / "manifest.json").read_text())["files"],
                             json.loads((second / "manifest.json").read_text())["files"])
            with self.assertRaises(FileExistsError):
                build(config, first)
            build(config, root / "partial", limit=1)
            with self.assertRaisesRegex(ValueError, "partial"):
                verify(root / "partial")

    def test_match_plan_disjoint_cost_main_reserve_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = root / "match.jsonl"
            rows = [{"game_hash": str(i), "sfen": f"board{i} b - 24", "ratings": [3500, 4000],
                     "eval": 0, "game_ply": 24} for i in range(12)]
            pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
            (root / "manifest.json").write_text(json.dumps({"complete": True,
                "files": {"match.jsonl": {"sha256": sha(pool)}}}))
            result = select(root, root / "selection", count=8, cost=2)
            self.assertEqual([v["count"] for v in result["files"].values()], [2, 8, 2])
            seen = set()
            for name in result["files"]:
                part = [json.loads(line) for line in (root / "selection" / name).read_text().splitlines()]
                ids = {r["game_hash"] for r in part}
                self.assertFalse(ids & seen)
                seen.update(ids)
            self.assertEqual(len(seen), 12)
            with self.assertRaisesRegex(ValueError, "insufficient"):
                select(root, root / "too-many", count=20)
            pool.write_text(pool.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "modified"):
                select(root, root / "tampered", count=8, cost=2)


if __name__ == "__main__":
    unittest.main()
