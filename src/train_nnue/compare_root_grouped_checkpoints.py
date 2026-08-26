"""Paired root-level comparison of trained routers on grouped validation leaves."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import features as nnue_features
from train_nnue.diagnose_gate import infer_model_configuration, per_record_loss
from train_nnue.evaluate_root_grouped_teachers import bootstrap_interval
from train_nnue.expert_blending_model import create_expert_blending_model
from train_nnue.root_grouped_dataset import RootGroupedDataset


def load_model(path, backbone_weights, nnue_checkpoint, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    configuration = infer_model_configuration(checkpoint["state_dict"])
    auxiliary_dim = int(
        checkpoint["state_dict"]["model.adapter.fc1.weight"].shape[1] - 192
    )
    model = create_expert_blending_model(
        backbone_weights_path=backbone_weights,
        nnue_ckpt_path=nnue_checkpoint,
        feature_set=nnue_features.get_feature_set_from_name("HalfKP"),
        n_experts=configuration["n_experts"],
        adapter_hidden=configuration["adapter_hidden"],
        adapter_auxiliary_dim=auxiliary_dim,
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
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint.get("hyper_parameters", {}), auxiliary_dim


def evaluate(path, args, teacher_weights=None):
    model, hparams, auxiliary_dim = load_model(
        path, args.backbone_weights, args.nnue_checkpoint, args.device
    )
    dataset = RootGroupedDataset(
        args.data,
        "HalfKP",
        args.root_batch_size,
        device=args.device,
        root_feature_cache_path=args.root_features,
    )
    losses = []
    tail_losses = []
    teacher_matches = []
    teacher_cross_entropies = []
    processed = 0
    iterator = iter(dataset)
    with torch.inference_mode():
        while processed < args.max_roots:
            batch = next(iterator)
            x1, x2, us, them, white, black, outcome, score, _ = batch[:9]
            auxiliary = batch[9] if len(batch) > 9 else None
            take = min(int(x1.shape[0]), args.max_roots - processed)
            gate = model.adapter(
                model.backbone(x1, x2),
                auxiliary=auxiliary if auxiliary_dim else None,
                training=False,
            )
            raw = model.nnue_experts(
                gate.repeat_interleave(dataset.group_size, dim=0),
                us,
                them,
                white,
                black,
            )
            per_leaf_loss = per_record_loss(
                raw,
                score,
                outcome,
                float(hparams.get("score_scaling", 361.0)),
                float(hparams.get("lambda_", 1.0)),
                float(hparams.get("label_smoothing_eps", 0.0)),
            ).reshape(int(x1.shape[0]), dataset.group_size)
            group_loss = per_leaf_loss.mean(dim=1)
            tail_count = max(1, int(np.ceil(dataset.group_size * args.tail_fraction)))
            tail_loss = per_leaf_loss.topk(tail_count, dim=1).values.mean(dim=1)
            losses.append(group_loss[:take].cpu().numpy())
            tail_losses.append(tail_loss[:take].cpu().numpy())
            if teacher_weights is not None:
                target = torch.from_numpy(
                    teacher_weights[processed : processed + take].copy()
                ).to(args.device)
                teacher_matches.append(
                    (gate[:take].argmax(dim=1) == target.argmax(dim=1))
                    .cpu()
                    .numpy()
                )
                teacher_cross_entropies.append(
                    (-(target * (gate[:take] + 1e-12).log()).sum(dim=1))
                    .cpu()
                    .numpy()
                )
            processed += take
    del model
    torch.cuda.empty_cache()
    result = {
        "losses": np.concatenate(losses),
        "tail_losses": np.concatenate(tail_losses),
    }
    if teacher_weights is not None:
        result["teacher_matches"] = np.concatenate(teacher_matches)
        result["teacher_cross_entropies"] = np.concatenate(
            teacher_cross_entropies
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--root-features")
    parser.add_argument(
        "--teacher",
        help=(
            "Optional shared-teacher cache. Omit it when only paired task-loss "
            "comparison is needed."
        ),
    )
    parser.add_argument("--backbone-weights", required=True)
    parser.add_argument("--nnue-checkpoint", required=True)
    parser.add_argument("--max-roots", type=int, required=True)
    parser.add_argument("--root-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=424204)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("checkpoint comparison requires CUDA")
    if not 0.0 < args.tail_fraction <= 1.0:
        parser.error("--tail-fraction must be in (0, 1]")
    teacher_weights = None
    if args.teacher:
        teacher = np.load(args.teacher, mmap_mode="r")
        teacher_weights = np.asarray(
            teacher[: args.max_roots, teacher.shape[1] // 2 :]
        )
    control = evaluate(args.control, args, teacher_weights)
    output = {
        "roots": args.max_roots,
        "control": {
            "checkpoint": str(Path(args.control).resolve()),
            "mean_group_loss": float(control["losses"].mean()),
            "mean_tail_loss": float(control["tail_losses"].mean()),
        },
        "candidates": [],
    }
    if teacher_weights is not None:
        output["control"].update(
            {
                "router_to_shared_teacher_top1_match": float(
                    control["teacher_matches"].mean()
                ),
                "shared_teacher_cross_entropy": float(
                    control["teacher_cross_entropies"].mean()
                ),
            }
        )
    for path in args.candidate:
        candidate = evaluate(path, args, teacher_weights)
        delta = candidate["losses"] - control["losses"]
        tail_delta = candidate["tail_losses"] - control["tail_losses"]
        candidate_output = {
            "checkpoint": str(Path(path).resolve()),
            "mean_group_loss": float(candidate["losses"].mean()),
            "paired_loss_delta_vs_control": {
                "mean": float(delta.mean()),
                "bootstrap_95_interval": bootstrap_interval(
                    delta, args.bootstrap_seed
                ),
                "fraction_improved": float((delta < 0.0).mean()),
            },
            "mean_tail_loss": float(candidate["tail_losses"].mean()),
            "paired_tail_loss_delta_vs_control": {
                "mean": float(tail_delta.mean()),
                "bootstrap_95_interval": bootstrap_interval(
                    tail_delta, args.bootstrap_seed
                ),
                "fraction_improved": float((tail_delta < 0.0).mean()),
            },
        }
        if teacher_weights is not None:
            candidate_output.update(
                {
                    "router_to_shared_teacher_top1_match": float(
                        candidate["teacher_matches"].mean()
                    ),
                    "shared_teacher_cross_entropy": float(
                        candidate["teacher_cross_entropies"].mean()
                    ),
                }
            )
        output["candidates"].append(candidate_output)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
