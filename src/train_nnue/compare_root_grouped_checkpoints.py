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
    model = create_expert_blending_model(
        backbone_weights_path=backbone_weights,
        nnue_ckpt_path=nnue_checkpoint,
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
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint.get("hyper_parameters", {})


def evaluate(path, args, teacher_weights=None):
    model, hparams = load_model(
        path, args.backbone_weights, args.nnue_checkpoint, args.device
    )
    dataset = RootGroupedDataset(
        args.data, "HalfKP", args.root_batch_size, device=args.device
    )
    losses = []
    teacher_matches = []
    teacher_cross_entropies = []
    processed = 0
    iterator = iter(dataset)
    with torch.inference_mode():
        while processed < args.max_roots:
            x1, x2, us, them, white, black, outcome, score, _ = next(iterator)
            take = min(int(x1.shape[0]), args.max_roots - processed)
            gate = model.adapter(model.backbone(x1, x2), training=False)
            raw = model.nnue_experts(
                gate.repeat_interleave(dataset.group_size, dim=0),
                us,
                them,
                white,
                black,
            )
            group_loss = per_record_loss(
                raw,
                score,
                outcome,
                float(hparams.get("score_scaling", 361.0)),
                float(hparams.get("lambda_", 1.0)),
                float(hparams.get("label_smoothing_eps", 0.0)),
            ).reshape(int(x1.shape[0]), dataset.group_size).mean(dim=1)
            losses.append(group_loss[:take].cpu().numpy())
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
    result = {"losses": np.concatenate(losses)}
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("checkpoint comparison requires CUDA")
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
