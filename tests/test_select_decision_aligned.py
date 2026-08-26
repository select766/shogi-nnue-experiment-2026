import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from select_decision_aligned_checkpoint import values_by_index  # noqa: E402


class SelectDecisionAlignedTest(unittest.TestCase):
    def test_reads_metric_values_in_epoch_order(self):
        curve = {"metrics": {"val_router_loss": [{"step": 2, "value": 1.2}, {"step": 4, "value": 1.1}]}}
        self.assertEqual([1.2, 1.1], values_by_index(curve, "val_router_loss"))


if __name__ == "__main__":
    unittest.main()
