import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from expand_adapter_input import FC1_WEIGHT, expand_adapter_input  # noqa: E402


class ExpandAdapterInputTest(unittest.TestCase):
    def test_zero_columns_preserve_linear_output(self):
        state = {FC1_WEIGHT: torch.randn(4, 3)}
        old = state[FC1_WEIGHT].clone()
        summary = expand_adapter_input(state, 2)
        self.assertEqual(5, summary["new_input_dim"])
        torch.testing.assert_close(state[FC1_WEIGHT][:, :3], old)
        torch.testing.assert_close(state[FC1_WEIGHT][:, 3:], torch.zeros(4, 2))


if __name__ == "__main__":
    unittest.main()
