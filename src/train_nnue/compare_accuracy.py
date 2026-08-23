"""Compare two best-move accuracy result files on the same positions."""

import argparse
import json
import os

from train_nnue.accuracy_statistics import (
    accuracy_summary,
    paired_summary,
    stratified_summary,
)
from train_nnue.eval_accuracy import load_dataset


def load_json(path):
    with open(path) as file:
        return json.load(file)


def aligned_matches(result, label):
    details = result.get("details")
    if not isinstance(details, list):
        raise ValueError(f"{label} result does not contain a details list")
    return [bool(detail["match"]) for detail in details]


def validate_alignment(baseline, candidate, records):
    baseline_details = baseline["details"]
    candidate_details = candidate["details"]
    if len(baseline_details) != len(candidate_details) or len(records) != len(
        candidate_details
    ):
        raise ValueError("baseline, candidate, and dataset lengths must match")
    for index, (base, cand, record) in enumerate(
        zip(baseline_details, candidate_details, records)
    ):
        expected = record["bestmove"]
        if (
            base.get("index") != index
            or cand.get("index") != index
            or base.get("sfen") != record["sfen"]
            or cand.get("sfen") != record["sfen"]
            or base.get("expected") != expected
            or cand.get("expected") != expected
        ):
            raise ValueError(f"results are not aligned with dataset at index {index}")


def validate_protocol(baseline, candidate):
    baseline_config = baseline.get("config", {})
    candidate_config = candidate.get("config", {})
    baseline_go = baseline_config.get("go_params")
    candidate_go = candidate_config.get("go_params")
    if not isinstance(baseline_go, dict) or not baseline_go:
        raise ValueError("baseline result does not record go_params")
    if baseline_go != candidate_go:
        raise ValueError("baseline and candidate use different go_params")
    for label, config in (("baseline", baseline_config), ("candidate", candidate_config)):
        threads = config.get("engine_options", {}).get("Threads")
        try:
            single_threaded = int(threads) == 1
        except (TypeError, ValueError):
            single_threaded = False
        if not single_threaded:
            raise ValueError(f"{label} result was not evaluated with Threads=1")
    return {"go_params": baseline_go, "threads": 1}


def compare_results(baseline, candidate, records):
    protocol = validate_protocol(baseline, candidate)
    validate_alignment(baseline, candidate, records)
    baseline_matches = aligned_matches(baseline, "baseline")
    candidate_matches = aligned_matches(candidate, "candidate")
    return {
        "protocol": protocol,
        "baseline": accuracy_summary(baseline_matches),
        "candidate": accuracy_summary(candidate_matches),
        "paired": paired_summary(baseline_matches, candidate_matches),
        "strata": stratified_summary(
            records, candidate_matches, baseline_matches=baseline_matches
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare paired accuracy results")
    parser.add_argument("--baseline", required=True, help="Baseline result JSON")
    parser.add_argument("--candidate", required=True, help="Candidate result JSON")
    parser.add_argument("--dataset", required=True, help="Shared JSONL dataset")
    parser.add_argument("--output", required=True, help="Comparison output JSON")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    records = load_dataset(args.dataset)
    comparison = compare_results(baseline, candidate, records)
    comparison["baseline_result_path"] = args.baseline
    comparison["candidate_result_path"] = args.candidate
    comparison["dataset_path"] = args.dataset

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(comparison, file, indent=2, ensure_ascii=False)

    paired = comparison["paired"]
    print(
        f"Baseline: {comparison['baseline']['accuracy']:.4f}; "
        f"candidate: {comparison['candidate']['accuracy']:.4f}; "
        f"difference: {paired['accuracy_difference']:+.4f}; "
        f"McNemar p={paired['exact_mcnemar_p_value']:.6g}"
    )
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
