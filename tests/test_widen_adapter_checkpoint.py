import sys
import unittest
from pathlib import Path


try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@unittest.skipIf(torch is None, "requires the nnue PyTorch environment")
class WidenAdapterCheckpointTest(unittest.TestCase):
    def test_widening_preserves_logits(self):
        from widen_adapter_checkpoint import widen_adapter_state_dict

        torch.manual_seed(5)
        state = {
            "model.adapter.fc1.weight": torch.randn(3, 4),
            "model.adapter.fc1.bias": torch.randn(3),
            "model.adapter.fc2.weight": torch.randn(2, 3),
            "model.adapter.fc2.bias": torch.randn(2),
        }
        inputs = torch.randn(7, 4)
        original = F.linear(
            F.relu(F.linear(inputs, state["model.adapter.fc1.weight"], state["model.adapter.fc1.bias"])),
            state["model.adapter.fc2.weight"],
            state["model.adapter.fc2.bias"],
        )
        summary = widen_adapter_state_dict(state, 12)
        widened = F.linear(
            F.relu(F.linear(inputs, state["model.adapter.fc1.weight"], state["model.adapter.fc1.bias"])),
            state["model.adapter.fc2.weight"],
            state["model.adapter.fc2.bias"],
        )
        torch.testing.assert_close(original, widened, atol=1e-6, rtol=1e-6)
        self.assertEqual(4, summary["repeats"])


if __name__ == "__main__":
    unittest.main()
