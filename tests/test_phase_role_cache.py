import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_phase_role_cache import roles_from_records  # noqa: E402


class PhaseRoleCacheTest(unittest.TestCase):
    def test_roles_are_fixed_32_ply_bins(self):
        records = np.zeros((6, 40), dtype=np.uint8)
        ply = [1, 32, 33, 224, 225, 400]
        for row, value in zip(records, ply):
            row[36:38] = np.frombuffer(value.to_bytes(2, "little"), dtype=np.uint8)
        np.testing.assert_array_equal(
            roles_from_records(records), [0, 0, 1, 6, 7, 7]
        )


if __name__ == "__main__":
    unittest.main()
