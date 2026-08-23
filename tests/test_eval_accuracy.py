import os
import unittest

from train_nnue.eval_accuracy import resolve_engine_option_paths


class ResolveEngineOptionPathsTest(unittest.TestCase):
    def test_resolves_all_repository_path_options(self):
        root = "/workspace/train-nnue"
        options = {
            "Threads": 1,
            "EvalDir": "bin/eval",
            "ExpertBlendingDir": "tmp/expert_blending_release",
        }

        resolved = resolve_engine_option_paths(options, root)

        self.assertEqual(resolved["Threads"], 1)
        self.assertEqual(resolved["EvalDir"], os.path.join(root, "bin/eval"))
        self.assertEqual(
            resolved["ExpertBlendingDir"],
            os.path.join(root, "tmp/expert_blending_release"),
        )
        self.assertEqual(options["EvalDir"], "bin/eval")

    def test_preserves_absolute_paths(self):
        options = {"EvalDir": "/models/eval"}

        resolved = resolve_engine_option_paths(options, "/workspace/train-nnue")

        self.assertEqual(resolved["EvalDir"], "/models/eval")


if __name__ == "__main__":
    unittest.main()
