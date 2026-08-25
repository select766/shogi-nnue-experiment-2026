import sys
import unittest
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@unittest.skipIf(torch is None, "requires the nnue PyTorch environment")
class ScaleExpertDeviationsTest(unittest.TestCase):
    def test_scale_preserves_mean_and_multiplies_deviation(self):
        from scale_expert_deviations import scale_state_dict

        original = torch.tensor([[1.0, 3.0], [3.0, 7.0]])
        state = {"model.nnue_experts.input_bias": original.clone()}
        summary = scale_state_dict(state, 2.0)
        expected = torch.tensor([[0.0, 1.0], [4.0, 9.0]])
        torch.testing.assert_close(expected, state["model.nnue_experts.input_bias"])
        torch.testing.assert_close(original.mean(0), state["model.nnue_experts.input_bias"].mean(0))
        self.assertEqual(1, len(summary["scaled_parameters"]))


if __name__ == "__main__":
    unittest.main()
