"""Evaluate whether an A-leaf shared teacher transfers to independent B leaves."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import features as nnue_features
from train_nnue.diagnose_gate import infer_model_configuration, per_record_loss
from train_nnue.expert_blending_model import create_expert_blending_model
from train_nnue.root_grouped_dataset import RootGroupedDataset


def bootstrap_interval(values, seed, samples=5000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(values, size=values.size, replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-b", required=True)
    parser.add_argument("--teacher-a", required=True)
    parser.add_argument("--teacher-b", required=True)
    parser.add_argument("--backbone-weights", required=True)
    parser.add_argument("--nnue-checkpoint", required=True)
    parser.add_argument("--max-roots", type=int, required=True)
    parser.add_argument("--root-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=424203)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("root-grouped teacher evaluation requires CUDA")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configuration = infer_model_configuration(checkpoint["state_dict"])
    hyperparameters = checkpoint.get("hyper_parameters", {})
    model = create_expert_blending_model(
        backbone_weights_path=args.backbone_weights,
        nnue_ckpt_path=args.nnue_checkpoint,
        feature_set=nnue_features.get_feature_set_from_name("HalfKP"),
        n_experts=configuration["n_experts"],
        adapter_hidden=configuration["adapter_hidden"],
        backbone_type=configuration["backbone_type"],
        blend_mode=configuration["blend_mode"],
        device="cpu",
    )
    model.load_state_dict(
        {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
    )
    model.requires_grad_(False)
    model.to(args.device).eval()
    n_experts = model.nnue_experts.n_experts
    teacher_a = np.load(args.teacher_a, mmap_mode="r")
    teacher_b = np.load(args.teacher_b, mmap_mode="r")
    for cache in (teacher_a, teacher_b):
        if cache.ndim != 2 or cache.shape[1] != 2 * n_experts:
            raise ValueError("teacher caches must contain expert losses and weights")
        if cache.shape[0] < args.max_roots:
            raise ValueError("teacher cache does not cover max-roots")
    dataset = RootGroupedDataset(
        args.data_b, "HalfKP", args.root_batch_size, device=args.device
    )
    if dataset.num_roots < args.max_roots:
        parser.error("data-b does not cover max-roots")
    group_size = dataset.group_size
    objective = {
        "score_scaling": float(hyperparameters.get("score_scaling", 361.0)),
        "lambda": float(hyperparameters.get("lambda_", 1.0)),
        "label_smoothing_eps": float(
            hyperparameters.get("label_smoothing_eps", 0.0)
        ),
    }
    collected = {"current": [], "teacher_a": [], "teacher_b": []}
    processed = 0
    iterator = iter(dataset)
    with torch.inference_mode():
        while processed < args.max_roots:
            x1, x2, us, them, white, black, outcome, score, _ = next(iterator)
            root_count = int(x1.shape[0])
            take = min(root_count, args.max_roots - processed)
            current = model.adapter(model.backbone(x1, x2), training=False)
            weights_a = torch.from_numpy(
                np.asarray(
                    teacher_a[processed : processed + root_count, n_experts:],
                    dtype=np.float32,
                ).copy()
            ).to(args.device)
            weights_b = torch.from_numpy(
                np.asarray(
                    teacher_b[processed : processed + root_count, n_experts:],
                    dtype=np.float32,
                ).copy()
            ).to(args.device)
            for name, weights in (
                ("current", current),
                ("teacher_a", weights_a),
                ("teacher_b", weights_b),
            ):
                raw = model.nnue_experts(
                    weights.repeat_interleave(group_size, dim=0),
                    us,
                    them,
                    white,
                    black,
                )
                losses = per_record_loss(
                    raw,
                    score,
                    outcome,
                    objective["score_scaling"],
                    objective["lambda"],
                    objective["label_smoothing_eps"],
                ).reshape(root_count, group_size).mean(dim=1)
                collected[name].append(losses[:take].cpu().numpy())
            processed += take
    losses = {name: np.concatenate(parts) for name, parts in collected.items()}
    weights_a = np.asarray(teacher_a[: args.max_roots, n_experts:])
    weights_b = np.asarray(teacher_b[: args.max_roots, n_experts:])
    mean_weights = 0.5 * (weights_a + weights_b)
    js = 0.5 * (
        (weights_a * (np.log(weights_a + 1e-12) - np.log(mean_weights + 1e-12))).sum(axis=1)
        + (weights_b * (np.log(weights_b + 1e-12) - np.log(mean_weights + 1e-12))).sum(axis=1)
    )
    delta = losses["teacher_a"] - losses["current"]
    current_argmax = []
    iterator = iter(dataset)
    processed = 0
    with torch.inference_mode():
        while processed < args.max_roots:
            x1, x2, *_ = next(iterator)
            take = min(int(x1.shape[0]), args.max_roots - processed)
            current_argmax.append(
                model.adapter(model.backbone(x1, x2), training=False)
                .argmax(dim=1)[:take]
                .cpu()
                .numpy()
            )
            processed += take
    current_argmax = np.concatenate(current_argmax)
    single_oracle_b = np.asarray(teacher_b[: args.max_roots, :n_experts]).argmin(axis=1)
    summary = {
        "roots": args.max_roots,
        "group_size_b": group_size,
        "mean_group_loss_b": {
            name: float(value.mean()) for name, value in losses.items()
        },
        "cross_set_loss_delta_a_minus_current": {
            "mean": float(delta.mean()),
            "bootstrap_95_interval": bootstrap_interval(
                delta, args.bootstrap_seed
            ),
            "fraction_improved": float((delta < 0.0).mean()),
        },
        "shared_teacher_argmax_agreement_a_b": float(
            (weights_a.argmax(axis=1) == weights_b.argmax(axis=1)).mean()
        ),
        "router_to_shared_teacher_top1_match": {
            "teacher_a": float((current_argmax == weights_a.argmax(axis=1)).mean()),
            "teacher_b": float((current_argmax == weights_b.argmax(axis=1)).mean()),
        },
        "router_to_single_expert_oracle_match_b": float(
            (current_argmax == single_oracle_b).mean()
        ),
        "teacher_distribution_js": {
            "mean": float(js.mean()),
            "p50": float(np.quantile(js, 0.5)),
            "p90": float(np.quantile(js, 0.9)),
            "p99": float(np.quantile(js, 0.99)),
        },
        "objective": objective,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
