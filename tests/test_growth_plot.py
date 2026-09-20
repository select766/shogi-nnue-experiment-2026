import json
from pathlib import Path
import tempfile
import unittest

from train_nnue.plot_growth import plot


class GrowthPlotTest(unittest.TestCase):
    def test_empty_then_measured_plot_csv_and_escaped_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "protocol.json").write_text(json.dumps({"config": {
                "series": "test-only", "baseline": {"name": "fixed"},
                "accuracy_nodes": 1000, "match_nodes": 1000}}))
            plot(directory)
            self.assertTrue((directory / "growth.png").read_bytes().startswith(b"\x89PNG"))
            measurements = directory / "measurements"
            measurements.mkdir()
            for i in range(2):
                row = {"finished_at": f"2026-09-{20+i}T00:00:00+00:00",
                       "champion": {"descriptor": {"id": "<script>test-only</script>"}},
                       "metrics": {"accuracy": {"accuracy": .6, "matches": 60, "total": 100,
                                                "wilson_95": {"lower": .5, "upper": .7}},
                                   "wins": 2, "losses": 1, "draws": 1,
                                   "win_rate": {"value": .5, "lower": .25, "upper": .75},
                                   "score_rate": {"value": .625, "lower": .3, "upper": .9}}}
                (measurements / f"{i}.json").write_text(json.dumps(row))
            plot(directory)
            self.assertEqual(len((directory / "history.csv").read_text().splitlines()), 3)
            self.assertIn("&lt;script&gt;", (directory / "growth.html").read_text())
            self.assertNotIn("<script>", (directory / "growth.html").read_text())
            self.assertTrue((directory / "growth.svg").is_file())


if __name__ == "__main__":
    unittest.main()
