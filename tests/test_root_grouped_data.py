import tempfile
import unittest
from pathlib import Path

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
