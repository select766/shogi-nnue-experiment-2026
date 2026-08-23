"""Diagnose Expert Blending routing and expert functional diversity."""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import features as nnue_features
from train_nnue.expert_blending_dataset import ExpertBlendingDataset
from train_nnue.expert_blending_model import (
    create_expert_blending_model,
    detect_blend_mode_from_state_dict,
)
from train_nnue.gate_diagnostics_statistics import compute_gate_diagnostics


def log(message):
    print(message, flush=True)


def per_record_loss(
    raw_value,
    score,
    outcome,
    score_scaling,
    lambda_,
    label_smoothing_eps,
):
    """Return the entropy-adjusted training objective for every position."""
    q = raw_value * 600.0 / score_scaling
    target = outcome * (1.0 - label_smoothing_eps * 2.0) + label_smoothing_eps
    teacher_probability = (score / score_scaling).sigmoid()
    epsilon = 1e-12
    teacher_entropy = -(
        teacher_probability * (teacher_probability + epsilon).log()
        + (1.0 - teacher_probability)
        * (1.0 - teacher_probability + epsilon).log()
    )
    outcome_entropy = -(
        target * (target + epsilon).log()
        + (1.0 - target) * (1.0 - target + epsilon).log()
    )
    teacher_loss = -(
        teacher_probability * F.logsigmoid(q)
        + (1.0 - teacher_probability) * F.logsigmoid(-q)
    )
    outcome_loss = -(
        target * F.logsigmoid(q)
        + (1.0 - target) * F.logsigmoid(-q)
    )
    loss = lambda_ * teacher_loss + (1.0 - lambda_) * outcome_loss
    entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
    return (loss - entropy).squeeze(-1)


def infer_model_configuration(state_dict):
    n_experts = int(state_dict["model.nnue_experts.input_weight"].shape[0])
    if "model.adapter.fc1.weight" in state_dict:
        backbone_type = "dnn"
        adapter_hidden = int(state_dict["model.adapter.fc1.weight"].shape[0])
    else:
        backbone_type = "nnue"
        adapter_hidden = None
    return {
        "n_experts": n_experts,
        "backbone_type": backbone_type,
        "adapter_hidden": adapter_hidden,
        "blend_mode": detect_blend_mode_from_state_dict(state_dict),
    }


def resolve_objective(args, checkpoint):
    hyper_parameters = checkpoint.get("hyper_parameters", {})

    def resolved(cli_value, key, default):
        return cli_value if cli_value is not None else hyper_parameters.get(key, default)

    return {
        "lambda": float(resolved(args.lambda_, "lambda_", 1.0)),
        "label_smoothing_eps": float(
            resolved(args.label_smoothing_eps, "label_smoothing_eps", 0.0)
        ),
        "score_scaling": float(
            resolved(args.score_scaling, "score_scaling", 361.0)
        ),
    }


def temperature_key(temperature):
    return format(float(temperature), ".12g")


def parse_temperatures(value):
    try:
        temperatures = [float(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("--temperatures must be comma-separated numbers") from error
    if not temperatures or any(value <= 0.0 for value in temperatures):
        raise ValueError("all temperatures must be positive")
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("temperatures must be unique")
    if 1.0 not in temperatures:
        temperatures.insert(0, 1.0)
    return temperatures


def collect_diagnostics(model, dataset, max_positions, objective, temperatures):
    n_experts = model.nnue_experts.n_experts
    collected = {
        "gate_weights": [],
        "expert_values": [],
        "expert_losses": [],
        "blended_losses": [],
        "teacher_scores": [],
        "game_plys": [],
        "temperature_gate_weights": {
            temperature_key(value): [] for value in temperatures
        },
        "temperature_blended_losses": {
            temperature_key(value): [] for value in temperatures
        },
    }
    processed = 0
    iterator = iter(dataset)
    with torch.inference_mode():
        while processed < max_positions:
            batch = next(iterator)
            if model.backbone_type == "nnue":
                (
                    us_bb,
                    them_bb,
                    white_bb,
                    black_bb,
                    us,
                    them,
                    white,
                    black,
                    outcome,
                    score,
                    ply,
                ) = batch
                gate_weights = model.backbone(
                    us_bb, them_bb, white_bb, black_bb, training=False
                )
            else:
                x1, x2, us, them, white, black, outcome, score, ply = batch
                features = model.backbone(x1, x2)
                gate_weights = model.adapter(features, training=False)

            if white.is_sparse:
                white = white.to_dense()
            if black.is_sparse:
                black = black.to_dense()
            temperature_weights = {}
            temperature_values = {}
            for temperature in temperatures:
                key = temperature_key(temperature)
                if temperature == 1.0:
                    weights_at_temperature = gate_weights
                else:
                    weights_at_temperature = torch.softmax(
                        torch.log(gate_weights.clamp_min(1e-30)) / temperature,
                        dim=-1,
                    )
                temperature_weights[key] = weights_at_temperature
                temperature_values[key] = model.nnue_experts(
                    weights_at_temperature, us, them, white, black
                )
            blended_value = temperature_values[temperature_key(1.0)]
            expert_values = torch.cat(
                [
                    model.nnue_experts.forward_expert(
                        expert, us, them, white, black
                    )
                    for expert in range(n_experts)
                ],
                dim=1,
            )
            expert_losses = torch.stack(
                [
                    per_record_loss(
                        expert_values[:, expert : expert + 1],
                        score,
                        outcome,
                        objective["score_scaling"],
                        objective["lambda"],
                        objective["label_smoothing_eps"],
                    )
                    for expert in range(n_experts)
                ],
                dim=1,
            )
            blended_losses = per_record_loss(
                blended_value,
                score,
                outcome,
                objective["score_scaling"],
                objective["lambda"],
                objective["label_smoothing_eps"],
            )
            temperature_losses = {}
            for temperature in temperatures:
                key = temperature_key(temperature)
                if temperature == 1.0:
                    temperature_losses[key] = blended_losses
                else:
                    temperature_losses[key] = per_record_loss(
                        temperature_values[key],
                        score,
                        outcome,
                        objective["score_scaling"],
                        objective["lambda"],
                        objective["label_smoothing_eps"],
                    )

            remaining = max_positions - processed
            take = min(int(gate_weights.shape[0]), remaining)
            collected["gate_weights"].append(gate_weights[:take].cpu().numpy())
            collected["expert_values"].append(
                (expert_values[:take] * 600.0).cpu().numpy()
            )
            collected["expert_losses"].append(
                expert_losses[:take].cpu().numpy()
            )
            collected["blended_losses"].append(
                blended_losses[:take].cpu().numpy()
            )
            collected["teacher_scores"].append(
                score[:take].squeeze(-1).cpu().numpy()
            )
            collected["game_plys"].append(
                ply[:take].squeeze(-1).cpu().numpy()
            )
            for temperature in temperatures:
                key = temperature_key(temperature)
                collected["temperature_gate_weights"][key].append(
                    temperature_weights[key][:take].cpu().numpy()
                )
                collected["temperature_blended_losses"][key].append(
                    temperature_losses[key][:take].cpu().numpy()
                )
            processed += take
            if processed % 1000 < take or processed == max_positions:
                log(f"Progress: {processed}/{max_positions}")

    output = {}
    for key, parts in collected.items():
        if isinstance(parts, dict):
            output[key] = {
                subkey: np.concatenate(subparts, axis=0)
                for subkey, subparts in parts.items()
            }
        else:
            output[key] = np.concatenate(parts, axis=0)
    return output


def build_output(args, configuration, objective, temperatures, collected):
    aggregate, per_position = compute_gate_diagnostics(
        collected["gate_weights"],
        collected["expert_values"],
        collected["expert_losses"],
        collected["blended_losses"],
    )
    temperature_sweep = []
    baseline_mean_loss = float(
        np.asarray(
            collected["temperature_blended_losses"][temperature_key(1.0)],
            dtype=np.float64,
        ).mean()
    )
    for temperature in temperatures:
        key = temperature_key(temperature)
        temperature_summary, _ = compute_gate_diagnostics(
            collected["temperature_gate_weights"][key],
            collected["expert_values"],
            collected["expert_losses"],
            collected["temperature_blended_losses"][key],
        )
        mean_loss = temperature_summary["routing"]["blended_mean_loss"]
        temperature_sweep.append(
            {
                "temperature": temperature,
                "mean_loss": mean_loss,
                "loss_delta_from_temperature_1": mean_loss - baseline_mean_loss,
                "gate": temperature_summary["gate"],
            }
        )

    details = []
    for index in range(aggregate["n_positions"]):
        details.append(
            {
                "index": index,
                "game_ply": int(collected["game_plys"][index]),
                "teacher_score": float(collected["teacher_scores"][index]),
                "gate_weights": collected["gate_weights"][index].tolist(),
                "entropy": float(per_position["entropy"][index]),
                "max_weight": float(per_position["max_weight"][index]),
                "top2_mass": float(per_position["top2_mass"][index]),
                "effective_experts": float(
                    per_position["effective_experts"][index]
                ),
                "gate_top1": int(per_position["gate_top1"][index]),
                "oracle_expert": int(per_position["oracle_expert"][index]),
                "gate_top1_matches_oracle": bool(
                    per_position["gate_top1_matches_oracle"][index]
                ),
                "blended_loss": float(per_position["blended_loss"][index]),
                "gate_top1_loss": float(per_position["gate_top1_loss"][index]),
                "oracle_loss": float(per_position["oracle_loss"][index]),
                "expert_value_variance": float(
                    per_position["expert_value_variance"][index]
                ),
                "expert_values": collected["expert_values"][index].tolist(),
                "expert_losses": collected["expert_losses"][index].tolist(),
                "temperature_losses": {
                    temperature_key(temperature): float(
                        collected["temperature_blended_losses"]
                        [temperature_key(temperature)][index]
                    )
                    for temperature in temperatures
                },
            }
        )
    return {
        "meta": {
            "checkpoint": args.checkpoint,
            "validation_dir": args.val,
            "backbone_weights": args.backbone_weights,
            "nnue_checkpoint": args.nnue_checkpoint,
            "feature_set": args.feature_set,
            "batch_size": args.batch_size,
            **configuration,
            "objective": objective,
            "temperatures": temperatures,
            "expert_values_unit": "centipawn-like NNUE output (raw value * 600)",
            "routing_semantics": (
                "DNN-side position selects an expert for the paired NNUE-side "
                "qsearch position, matching training."
            ),
        },
        "summary": aggregate,
        "temperature_sweep": temperature_sweep,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose Expert Blending gate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--backbone-weights")
    parser.add_argument("--nnue-checkpoint", required=True)
    parser.add_argument("--feature-set", default="HalfKP")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-positions", type=int, default=10000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lambda", type=float, dest="lambda_")
    parser.add_argument("--label-smoothing-eps", type=float)
    parser.add_argument("--score-scaling", type=float)
    parser.add_argument(
        "--gate-transform",
        choices=["softmax", "entmax15"],
        help="Override the checkpoint gate transform for a no-retraining diagnostic",
    )
    parser.add_argument(
        "--temperatures",
        default="1.0",
        help="Comma-separated inference temperatures (1.0 is always included)",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.max_positions <= 0:
        parser.error("--batch-size and --max-positions must be positive")
    try:
        temperatures = parse_temperatures(args.temperatures)
    except ValueError as error:
        parser.error(str(error))

    required = [args.checkpoint, args.val, args.nnue_checkpoint]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("gate diagnosis requires CUDA; use scripts/gpu_python.sh")

    log(f"Device: {args.device}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configuration = infer_model_configuration(checkpoint["state_dict"])
    configuration["gate_transform"] = args.gate_transform or checkpoint.get(
        "hyper_parameters", {}
    ).get("gate_transform", "softmax")
    if configuration["backbone_type"] == "dnn" and not args.backbone_weights:
        raise ValueError("--backbone-weights is required for a DNN gate")
    objective = resolve_objective(args, checkpoint)
    log(f"Configuration: {configuration}")
    log(f"Objective: {objective}")

    feature_set = nnue_features.get_feature_set_from_name(args.feature_set)
    model = create_expert_blending_model(
        backbone_weights_path=args.backbone_weights,
        nnue_ckpt_path=args.nnue_checkpoint,
        feature_set=feature_set,
        n_experts=configuration["n_experts"],
        adapter_hidden=configuration["adapter_hidden"] or 128,
        backbone_type=configuration["backbone_type"],
        blend_mode=configuration["blend_mode"],
        gate_transform=configuration["gate_transform"],
        device="cpu",
    )
    model_state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state)
    model.to(args.device)
    model.eval()

    dataset = ExpertBlendingDataset(
        os.path.abspath(args.val),
        args.feature_set,
        args.batch_size,
        device=args.device,
        shuffle=False,
        backbone_type=configuration["backbone_type"],
    )
    if args.max_positions > dataset.num_records:
        parser.error(
            f"--max-positions ({args.max_positions}) exceeds unique validation "
            f"records ({dataset.num_records})"
        )
    collected = collect_diagnostics(
        model, dataset, args.max_positions, objective, temperatures
    )
    output = build_output(
        args, configuration, objective, temperatures, collected
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary_output = f"{args.output}.tmp"
    with open(temporary_output, "w") as file:
        json.dump(output, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    os.replace(temporary_output, args.output)
    log(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
