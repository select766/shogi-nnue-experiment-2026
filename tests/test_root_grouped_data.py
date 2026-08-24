import tempfile
import unittest
from pathlib import Path

import numpy as np
import cshogi

from scripts.collect_search_leaf_groups import endpoint_record, record_to_sfen
from scripts.filter_root_grouped_validation import root_sfens
from scripts.split_grouped_paired_bin import packed_game_ply


class RootGroupedDataTest(unittest.TestCase):
    def test_reads_little_endian_game_ply(self):
        records = np.zeros((3, 40), dtype=np.uint8)
        records[:, 36] = [1, 0, 255]
        records[:, 37] = [0, 1, 1]
        np.testing.assert_array_equal(
            packed_game_ply(records), np.array([1, 256, 511])
        )

    def test_root_sfen_ignores_non_position_record_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roots.bin"
            first = bytes(range(32)) + bytes(8)
            second = bytes(reversed(range(32))) + bytes([9] * 8)
            path.write_bytes(first + second)
            self.assertEqual(root_sfens(path), [first[:32], second[:32]])

    def test_search_root_restores_game_ply_outside_packed_sfen(self):
        board = cshogi.Board()
        packed = np.zeros(1, dtype=cshogi.PackedSfen)
        board.to_psfen(packed)
        record = bytearray(40)
        record[:32] = packed[0]["sfen"].tobytes()
        record[36:38] = (123).to_bytes(2, "little")
        decoded = cshogi.Board()
        scratch = np.zeros(1, dtype=cshogi.PackedSfen)
        self.assertTrue(record_to_sfen(record, decoded, scratch).endswith(" 123"))

    def test_qsearch_endpoint_score_is_relative_to_endpoint_turn(self):
        record = endpoint_record(cshogi.Board().sfen(), 250, ["7g7f"])
        score = int.from_bytes(record[32:34], "little", signed=True)
        ply = int.from_bytes(record[36:38], "little")
        self.assertEqual(score, -250)
        self.assertEqual(ply, 2)


if __name__ == "__main__":
    unittest.main()
