"""Pure NumPy summaries for Expert Blending gate diagnostics."""

import numpy as np


def apply_gate_temperature(gate_weights, temperature):
    """Apply softmax temperature to already normalized gate probabilities."""
    gate_weights = np.asarray(gate_weights, dtype=np.float64)
    if gate_weights.ndim != 2:
        raise ValueError("gate_weights must be a two-dimensional array")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if np.any(gate_weights < 0.0) or not np.allclose(
        gate_weights.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("gate weights must be non-negative and sum to one")
    if temperature == 1.0:
        return gate_weights.copy()
    logits = np.log(np.clip(gate_weights, 1e-300, None)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    unnormalized = np.exp(logits)
    return unnormalized / unnormalized.sum(axis=1, keepdims=True)


def distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _correlation_matrix(values):
    """Return a JSON-safe column-wise Pearson correlation matrix."""
    values = np.asarray(values, dtype=np.float64)
    n_columns = values.shape[1]
    matrix = []
    finite_off_diagonal = []
    for left in range(n_columns):
        row = []
        left_values = values[:, left]
        for right in range(n_columns):
            right_values = values[:, right]
            if left_values.std() == 0.0 or right_values.std() == 0.0:
                correlation = None
            else:
                correlation = float(np.corrcoef(left_values, right_values)[0, 1])
            row.append(correlation)
            if left < right and correlation is not None:
                finite_off_diagonal.append(correlation)
        matrix.append(row)
    return matrix, distribution_summary(finite_off_diagonal)


def compute_gate_diagnostics(
    gate_weights,
    expert_values,
    expert_losses,
    blended_losses,
):
    """Compute per-position arrays and aggregate gate/expert diagnostics."""
    gate_weights = np.asarray(gate_weights, dtype=np.float64)
    expert_values = np.asarray(expert_values, dtype=np.float64)
    expert_losses = np.asarray(expert_losses, dtype=np.float64)
    blended_losses = np.asarray(blended_losses, dtype=np.float64).reshape(-1)

    if gate_weights.ndim != 2 or gate_weights.shape[1] == 0:
        raise ValueError("gate_weights must have shape (positions, experts)")
    if expert_values.shape != gate_weights.shape:
        raise ValueError("expert_values shape must match gate_weights")
    if expert_losses.shape != gate_weights.shape:
        raise ValueError("expert_losses shape must match gate_weights")
    if blended_losses.shape[0] != gate_weights.shape[0]:
        raise ValueError("blended_losses length must match gate_weights")
    if np.any(gate_weights < 0.0) or not np.allclose(
        gate_weights.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("gate weights must be non-negative and sum to one")

    n_positions, n_experts = gate_weights.shape
    safe_weights = np.clip(gate_weights, 1e-300, None)
    entropy = -(gate_weights * np.log(safe_weights)).sum(axis=1)
    max_weight = gate_weights.max(axis=1)
    if n_experts == 1:
        top2_mass = max_weight.copy()
    else:
        top2_mass = np.partition(gate_weights, -2, axis=1)[:, -2:].sum(axis=1)
    effective_experts = np.exp(entropy)
    dominant = gate_weights.argmax(axis=1)
    oracle = expert_losses.argmin(axis=1)
    indices = np.arange(n_positions)
    top1_losses = expert_losses[indices, dominant]
    oracle_losses = expert_losses[indices, oracle]

    utilization_counts = np.bincount(dominant, minlength=n_experts)
    utilization_rates = utilization_counts / n_positions
    oracle_counts = np.bincount(oracle, minlength=n_experts)
    oracle_rates = oracle_counts / n_positions
    routing_confusion = np.zeros((n_experts, n_experts), dtype=np.int64)
    np.add.at(routing_confusion, (dominant, oracle), 1)
    expected_match_rate = float(np.dot(utilization_rates, oracle_rates))
    mean_weights = gate_weights.mean(axis=0)
    uniform = 1.0 / n_experts
    positive_mean_weights = mean_weights > 0.0
    mean_weight_kl = float(
        np.sum(
            mean_weights[positive_mean_weights]
            * np.log(mean_weights[positive_mean_weights] / uniform)
        )
    )
    value_correlations, off_diagonal_summary = _correlation_matrix(expert_values)
    per_position_variance = expert_values.var(axis=1)
    mean_expert_losses = expert_losses.mean(axis=0)
    best_fixed_expert = int(mean_expert_losses.argmin())
    best_fixed_loss = float(mean_expert_losses[best_fixed_expert])
    oracle_mean_loss = float(oracle_losses.mean())

    aggregate = {
        "n_positions": n_positions,
        "n_experts": n_experts,
        "gate": {
            "entropy": distribution_summary(entropy),
            "max_weight": distribution_summary(max_weight),
            "top2_mass": distribution_summary(top2_mass),
            "effective_experts": distribution_summary(effective_experts),
            "argmax_utilization_counts": utilization_counts.astype(int).tolist(),
            "argmax_utilization_rates": utilization_rates.tolist(),
            "argmax_utilization_cv": float(
                utilization_rates.std() / utilization_rates.mean()
            ),
            "mean_weights": mean_weights.tolist(),
            "mean_weight_cv": float(mean_weights.std() / mean_weights.mean()),
            "mean_weight_kl_from_uniform": mean_weight_kl,
            "dead_experts_by_argmax": np.flatnonzero(utilization_counts == 0)
            .astype(int)
            .tolist(),
        },
        "expert_function": {
            "value_correlation_matrix": value_correlations,
            "off_diagonal_correlation": off_diagonal_summary,
            "per_position_value_variance": distribution_summary(
                per_position_variance
            ),
        },
        "routing": {
            "blended_mean_loss": float(blended_losses.mean()),
            "gate_top1_mean_loss": float(top1_losses.mean()),
            "oracle_mean_loss": oracle_mean_loss,
            "oracle_improvement_over_blended": float(
                blended_losses.mean() - oracle_mean_loss
            ),
            "oracle_improvement_over_gate_top1": float(
                top1_losses.mean() - oracle_mean_loss
            ),
            "gate_top1_oracle_match_rate": float((dominant == oracle).mean()),
            "gate_top1_oracle_expected_match_rate_from_marginals": (
                expected_match_rate
            ),
            "gate_top1_oracle_match_lift": (
                float((dominant == oracle).mean() / expected_match_rate)
                if expected_match_rate > 0.0
                else None
            ),
            "oracle_expert_counts": oracle_counts.astype(int).tolist(),
            "oracle_expert_rates": oracle_rates.tolist(),
            "gate_top1_by_oracle_confusion": routing_confusion.tolist(),
            "best_fixed_expert": best_fixed_expert,
            "best_fixed_expert_mean_loss": best_fixed_loss,
            "oracle_improvement_over_best_fixed": best_fixed_loss
            - oracle_mean_loss,
            "per_expert_mean_loss": mean_expert_losses.tolist(),
        },
    }
    per_position = {
        "entropy": entropy,
        "max_weight": max_weight,
        "top2_mass": top2_mass,
        "effective_experts": effective_experts,
        "gate_top1": dominant,
        "oracle_expert": oracle,
        "gate_top1_matches_oracle": dominant == oracle,
        "blended_loss": blended_losses,
        "gate_top1_loss": top1_losses,
        "oracle_loss": oracle_losses,
        "expert_value_variance": per_position_variance,
    }
    return aggregate, per_position
