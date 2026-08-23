import unittest

import numpy as np

from train_nnue.gate_diagnostics_statistics import (
    apply_gate_temperature,
    compute_gate_diagnostics,
)


class GateDiagnosticsStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.gates = np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4]])
        self.values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        self.losses = np.array([[0.1, 0.2], [0.4, 0.3], [0.5, 0.1]])
        self.blended = np.array([0.15, 0.35, 0.30])

    def test_gate_and_routing_summary(self):
        summary, per_position = compute_gate_diagnostics(
            self.gates, self.values, self.losses, self.blended
        )

        self.assertEqual(summary["gate"]["argmax_utilization_counts"], [2, 1])
        self.assertAlmostEqual(summary["gate"]["argmax_utilization_cv"], 1 / 3)
        self.assertAlmostEqual(summary["gate"]["mean_weight_kl_from_uniform"], 0.0)
        self.assertEqual(summary["gate"]["dead_experts_by_argmax"], [])
        self.assertAlmostEqual(
            summary["routing"]["gate_top1_oracle_match_rate"], 2 / 3
        )
        self.assertEqual(summary["routing"]["oracle_expert_counts"], [1, 2])
        self.assertEqual(
            summary["routing"]["gate_top1_by_oracle_confusion"],
            [[1, 1], [0, 1]],
        )
        self.assertAlmostEqual(
            summary["routing"]["oracle_improvement_over_blended"], 0.1
        )
        self.assertEqual(summary["routing"]["best_fixed_expert"], 1)
        np.testing.assert_array_equal(per_position["gate_top1"], [0, 1, 0])
        np.testing.assert_array_equal(per_position["oracle_expert"], [0, 1, 1])

    def test_per_position_sparsity_and_expert_similarity(self):
        summary, per_position = compute_gate_diagnostics(
            self.gates, self.values, self.losses, self.blended
        )

        np.testing.assert_allclose(per_position["top2_mass"], 1.0)
        np.testing.assert_allclose(
            per_position["effective_experts"], np.exp(per_position["entropy"])
        )
        self.assertAlmostEqual(
            summary["expert_function"]["value_correlation_matrix"][0][1], 1.0
        )
        self.assertAlmostEqual(
            summary["expert_function"]["per_position_value_variance"]["mean"],
            (0.25 + 1.0 + 2.25) / 3,
        )

    def test_rejects_invalid_probabilities(self):
        invalid = self.gates.copy()
        invalid[0] = [0.8, 0.3]

        with self.assertRaisesRegex(ValueError, "sum to one"):
            compute_gate_diagnostics(
                invalid, self.values, self.losses, self.blended
            )

    def test_constant_expert_correlation_is_json_safe(self):
        values = np.ones_like(self.values)

        summary, _ = compute_gate_diagnostics(
            self.gates, values, self.losses, self.blended
        )

        self.assertIsNone(
            summary["expert_function"]["value_correlation_matrix"][0][1]
        )
        self.assertIsNone(
            summary["expert_function"]["off_diagonal_correlation"]
        )

    def test_temperature_sharpens_without_changing_argmax(self):
        sharpened = apply_gate_temperature(self.gates, 0.5)

        np.testing.assert_array_equal(
            sharpened.argmax(axis=1), self.gates.argmax(axis=1)
        )
        self.assertTrue(np.all(sharpened.max(axis=1) > self.gates.max(axis=1)))
        np.testing.assert_allclose(sharpened.sum(axis=1), 1.0)

    def test_temperature_one_is_identity(self):
        np.testing.assert_allclose(
            apply_gate_temperature(self.gates, 1.0), self.gates
        )

    def test_rejects_non_positive_temperature(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            apply_gate_temperature(self.gates, 0.0)


if __name__ == "__main__":
    unittest.main()
