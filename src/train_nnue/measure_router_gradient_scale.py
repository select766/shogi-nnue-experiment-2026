"""Measure adapter gradient norms for task and router-distillation losses."""

import argparse

import torch

import features as nnue_features
from train_nnue.diagnose_gate import infer_model_configuration, per_record_loss
from train_nnue.expert_blending_dataset import ExpertBlendingDataset
from train_nnue.expert_blending_model import create_expert_blending_model
from train_nnue.train_expert_blending import (
    compute_router_statistics,
    router_teacher_distribution,
)


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
    return torch.sqrt(sum(gradient.square().sum() for gradient in gradients))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bin-dir", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--backbone-weights", required=True)
    parser.add_argument("--nnue-checkpoint", required=True)
    parser.add_argument(
        "--teacher-mode", choices=["hard", "soft", "cached"], required=True
    )
    parser.add_argument("--temperature", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configuration = infer_model_configuration(checkpoint["state_dict"])
    hyperparameters = checkpoint.get("hyper_parameters", {})
    feature_set = nnue_features.get_feature_set_from_name("HalfKP")
    model = create_expert_blending_model(
        backbone_weights_path=args.backbone_weights,
        nnue_ckpt_path=args.nnue_checkpoint,
        feature_set=feature_set,
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
    model.nnue_experts.requires_grad_(False)
    model.to(args.device).train()
    dataset = ExpertBlendingDataset(
        args.bin_dir,
        "HalfKP",
        args.batch_size,
        device=args.device,
        shuffle=False,
        teacher_cache_path=args.teacher_cache,
    )
    x1, x2, us, them, white, black, outcome, score, _, teacher_data = next(
        iter(dataset)
    )
    if args.teacher_mode == "cached":
        n_experts = model.nnue_experts.n_experts
        expert_losses = teacher_data[:, :n_experts]
        teacher_weights = teacher_data[:, n_experts:]
    else:
        expert_losses = teacher_data
        teacher_weights = router_teacher_distribution(
            expert_losses, args.teacher_mode, args.temperature
        )
    raw_value, gate_weights = model(
        x1, x2, us, them, white, black, training=False, return_gate=True
    )
    task_loss = per_record_loss(
        raw_value,
        score,
        outcome,
        float(hyperparameters.get("score_scaling", 361.0)),
        float(hyperparameters.get("lambda_", 1.0)),
        float(hyperparameters.get("label_smoothing_eps", 0.0)),
    ).mean()
    router_loss = compute_router_statistics(
        gate_weights, expert_losses, teacher_weights
    )["loss"]
    parameters = list(model.adapter.parameters())
    task_norm = gradient_norm(task_loss, parameters)
    router_norm = gradient_norm(router_loss, parameters)
    ratio = task_norm / router_norm
    print(f"task_loss={task_loss.item():.9f} task_grad_norm={task_norm.item():.9f}")
    print(
        f"router_loss={router_loss.item():.9f} "
        f"router_grad_norm={router_norm.item():.9f}"
    )
    print(f"lambda_for_0.25x={(0.25 * ratio).item():.9f}")
    print(f"lambda_for_1.00x={ratio.item():.9f}")


if __name__ == "__main__":
    main()
