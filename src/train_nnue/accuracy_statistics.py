"""Statistics and stratified summaries for paired move-accuracy evaluation."""

import math

import cshogi


def wilson_interval(matches, total, z=1.959963984540054):
    """Return the two-sided Wilson score interval for a binomial proportion."""
    if total == 0:
        return None
    proportion = matches / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {"lower": center - radius, "upper": center + radius}


def exact_mcnemar_p_value(baseline_only, candidate_only):
    """Return the exact two-sided McNemar p-value.

    Conditional on the number of discordant pairs, either direction has
    probability 1/2 under the null hypothesis.
    """
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, candidate_only)
    log_denominator = discordant * math.log(2.0)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - log_denominator
        for k in range(tail + 1)
    ]
    largest = max(log_terms)
    one_sided = math.exp(largest) * sum(math.exp(term - largest) for term in log_terms)
    return min(1.0, 2.0 * one_sided)


def accuracy_summary(matches):
    """Summarize a sequence of boolean correctness values."""
    total = len(matches)
    count = sum(bool(value) for value in matches)
    return {
        "accuracy": count / total if total else None,
        "matches": count,
        "total": total,
        "wilson_95": wilson_interval(count, total),
    }


def paired_summary(baseline_matches, candidate_matches):
    """Summarize paired correctness values for two models."""
    if len(baseline_matches) != len(candidate_matches):
        raise ValueError("paired results have different lengths")
    both_correct = baseline_only = candidate_only = both_wrong = 0
    for baseline, candidate in zip(baseline_matches, candidate_matches):
        if baseline and candidate:
            both_correct += 1
        elif baseline:
            baseline_only += 1
        elif candidate:
            candidate_only += 1
        else:
            both_wrong += 1
    total = len(baseline_matches)
    return {
        "both_correct": both_correct,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "both_wrong": both_wrong,
        "discordant": baseline_only + candidate_only,
        "accuracy_difference": (
            (candidate_only - baseline_only) / total if total else None
        ),
        "exact_mcnemar_p_value": exact_mcnemar_p_value(
            baseline_only, candidate_only
        ),
    }


def _game_ply(record):
    value = record.get("game_ply", record.get("ply"))
    if value is None:
        return None
    return int(value)


def _has_entering_king(sfen):
    board = cshogi.Board(sfen)
    black_king = board.king_square(cshogi.BLACK)
    white_king = board.king_square(cshogi.WHITE)
    return (
        black_king >= 0 and cshogi.make_rank(black_king) <= 2
    ) or (
        white_king >= 0 and cshogi.make_rank(white_king) >= 6
    )


def stratum_labels(record):
    """Return stable evaluation strata for one dataset record."""
    labels = {}
    game_ply = _game_ply(record)
    if game_ply is not None:
        if game_ply <= 40:
            labels["game_ply"] = "early_1_40"
        elif game_ply <= 100:
            labels["game_ply"] = "middle_41_100"
        else:
            labels["game_ply"] = "late_101_plus"

    if record.get("eval") is not None:
        absolute_eval = abs(int(record["eval"]))
        if absolute_eval <= 500:
            labels["absolute_teacher_eval"] = "even_0_500"
        elif absolute_eval <= 2000:
            labels["absolute_teacher_eval"] = "advantage_501_2000"
        else:
            labels["absolute_teacher_eval"] = "winning_2001_plus"

    labels["entering_king"] = (
        "entering_king" if _has_entering_king(record["sfen"]) else "other"
    )
    return labels


def stratified_summary(records, candidate_matches, baseline_matches=None):
    """Aggregate accuracy, and optionally paired statistics, by strata."""
    if len(records) != len(candidate_matches):
        raise ValueError("dataset and candidate result lengths differ")
    if baseline_matches is not None and len(records) != len(baseline_matches):
        raise ValueError("dataset and baseline result lengths differ")

    groups = {}
    for index, record in enumerate(records):
        for dimension, label in stratum_labels(record).items():
            group = groups.setdefault(dimension, {}).setdefault(
                label, {"candidate": [], "baseline": []}
            )
            group["candidate"].append(candidate_matches[index])
            if baseline_matches is not None:
                group["baseline"].append(baseline_matches[index])

    output = {}
    for dimension in ("game_ply", "absolute_teacher_eval", "entering_king"):
        dimension_groups = groups.get(dimension)
        if not dimension_groups:
            output[dimension] = {
                "available": False,
                "reason": (
                    "dataset records do not contain game_ply or ply"
                    if dimension == "game_ply"
                    else "required dataset field is unavailable"
                ),
            }
            continue
        values = {}
        for label, group in dimension_groups.items():
            value = {"candidate": accuracy_summary(group["candidate"])}
            if baseline_matches is not None:
                value["baseline"] = accuracy_summary(group["baseline"])
                value["paired"] = paired_summary(
                    group["baseline"], group["candidate"]
                )
            values[label] = value
        output[dimension] = {"available": True, "groups": values}
    return output
