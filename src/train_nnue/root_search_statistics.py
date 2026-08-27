"""Fixed shallow-MultiPV statistics shared by cache and online evaluation."""

from __future__ import annotations

import math

import numpy as np


FEATURE_NAMES = [
    "top1_score",
    "top1_top2_margin",
    "top1_last_width",
    "score_standard_deviation",
    "score_entropy",
    "multipv_fraction",
]


def parse_usi_multipv_score(line):
    fields = line.split()
    if not fields or fields[0] != "info" or "score" not in fields:
        return None
    score_index = fields.index("score")
    if score_index + 2 >= len(fields):
        return None
    kind, token = fields[score_index + 1 : score_index + 3]
    try:
        value = int(token)
    except ValueError:
        return None
    if kind == "mate":
        sign = -1 if token.startswith("-") else 1
        score = sign * (32000 - min(abs(value), 1000))
    elif kind == "cp":
        score = max(-32000, min(32000, value))
    else:
        return None
    multipv = 1
    if "multipv" in fields:
        index = fields.index("multipv")
        if index + 1 >= len(fields):
            return None
        try:
            multipv = int(fields[index + 1])
        except ValueError:
            return None
    if multipv < 1:
        return None
    return multipv, float(score)


def statistics_from_scores(scores):
    ordered = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if not 1 <= len(ordered) <= 4 or not np.all(np.isfinite(ordered)):
        raise ValueError("expected one to four finite MultiPV scores")
    top = ordered[0]
    margin = top - ordered[1] if len(ordered) > 1 else 0.0
    width = top - ordered[-1]
    standard_deviation = ordered.std()
    if len(ordered) == 1:
        entropy = 0.0
    else:
        logits = ordered / 200.0
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        entropy = float(
            -(probabilities * np.log(probabilities + 1e-12)).sum()
            / math.log(len(ordered))
        )
    return np.asarray(
        [
            np.clip(top / 2000.0, -1.0, 1.0),
            np.clip(margin / 1000.0, 0.0, 1.0),
            np.clip(width / 1000.0, 0.0, 1.0),
            np.clip(standard_deviation / 1000.0, 0.0, 1.0),
            np.clip(entropy, 0.0, 1.0),
            len(ordered) / 4.0,
        ],
        dtype=np.float32,
    )


def shallow_multipv_features(engine, sfen, nodes=1024, multipv=4):
    engine.send("setoption name Clear Hash")
    engine.send(f"setoption name MultiPV value {multipv}")
    engine.send(f"position sfen {sfen}")
    engine.send(f"go nodes {nodes}")
    latest = {}
    while True:
        line = engine.read_until(lambda _: True)
        parsed = parse_usi_multipv_score(line)
        if parsed is not None:
            index, score = parsed
            if index <= multipv:
                latest[index] = score
        if line.startswith("bestmove "):
            break
    if not latest:
        raise ValueError(f"shallow search returned no MultiPV score: {sfen}")
    return statistics_from_scores(list(latest.values()))
