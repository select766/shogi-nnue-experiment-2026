import unittest
from unittest.mock import patch

from train_nnue import compare_root_grouped_checkpoints as module


class CompareRootFeatureCachesTest(unittest.TestCase):
    def test_evaluate_uses_explicit_cache(self):
        # Signature-level regression: different control/candidate caches are supported.
        self.assertIn("root_features", module.evaluate.__code__.co_varnames)


if __name__ == "__main__":
    unittest.main()
