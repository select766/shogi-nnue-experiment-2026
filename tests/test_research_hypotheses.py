import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_research_hypotheses import validate_registry  # noqa: E402


class ResearchHypothesisRegistryTest(unittest.TestCase):
    def test_registry_is_consistent_and_covers_all_results(self):
        self.assertEqual([], validate_registry())


if __name__ == "__main__":
    unittest.main()
