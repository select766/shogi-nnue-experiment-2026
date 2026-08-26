"""
Expert Blending モデルの学習スクリプト。

PyTorch Lightning ベースで、既存 NNUE と同じ損失関数
(teacher_loss + outcome_loss の λ ブレンド) を使用する。
勾配は DNN_adapter と NNUE_weights にのみ流す (backbone は frozen)。

Usage:
    scripts/gpu_python.sh -m train_nnue.train_expert_blending \
        --train dataset/split_v1_paired_uniform_50/train \
        --val dataset/split_v1_paired_uniform_50/val1 \
        --backbone-weights tmp/dlshogi-model/model_resnet10_swish-072 \
        --nnue-checkpoint logs/halfkp_v1/checkpoints/83000.ckpt
"""

import argparse
import math
import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers

import features as nnue_features
from train_nnue.expert_blending_dataset import (
    create_data_loaders,
)
from train_nnue.expert_blending_model import create_expert_blending_model
from train_nnue.root_grouped_dataset import create_root_grouped_loaders


def compute_gate_statistics(gate_weights):
    """Return differentiable sparsity and balance statistics for a batch."""
    entropy_per_position = -(
        gate_weights * (gate_weights + 1e-12).log()
    ).sum(dim=-1)
    mean_gate = gate_weights.mean(dim=0)
    balance_kl = (
        mean_gate * (mean_gate * gate_weights.shape[1] + 1e-12).log()
    ).sum()
    return {
        "entropy": entropy_per_position.mean(),
        "balance_kl": balance_kl,
        "effective_experts": entropy_per_position.exp().mean(),
        "max_weight": gate_weights.max(dim=-1).values.mean(),
        "top2_mass": gate_weights.topk(
            min(2, gate_weights.shape[1]), dim=-1
        ).values.sum(dim=-1).mean(),
    }


def router_teacher_distribution(expert_losses, mode, temperature=0.005):
    """Build detached hard or loss-temperature teacher probabilities."""
    if expert_losses.ndim != 2:
        raise ValueError("expert_losses must have shape (positions, experts)")
    if mode == "hard":
        return F.one_hot(
            expert_losses.argmin(dim=-1), num_classes=expert_losses.shape[1]
        ).to(dtype=expert_losses.dtype)
    if mode == "soft":
        if temperature <= 0.0:
            raise ValueError("router teacher temperature must be positive")
        centered = expert_losses - expert_losses.min(dim=-1, keepdim=True).values
        return torch.softmax(-centered / temperature, dim=-1)
    raise ValueError(f"Unsupported router teacher mode: {mode}")


def compute_router_statistics(gate_weights, expert_losses, teacher_weights):
    """Return router supervision loss and oracle-alignment metrics."""
    epsilon = 1e-12
    router_loss = -(
        teacher_weights * (gate_weights + epsilon).log()
    ).sum(dim=-1).mean()
    gate_top1 = gate_weights.argmax(dim=-1)
    oracle = expert_losses.argmin(dim=-1)
    indices = torch.arange(gate_weights.shape[0], device=gate_weights.device)
    regret = (
        expert_losses[indices, gate_top1] - expert_losses[indices, oracle]
    ).mean()
    match = (gate_top1 == oracle).to(dtype=gate_weights.dtype).mean()
    teacher_match = (
        gate_top1 == teacher_weights.argmax(dim=-1)
    ).to(dtype=gate_weights.dtype).mean()
    expected_match = (
        F.one_hot(gate_top1, num_classes=gate_weights.shape[1])
        .to(dtype=gate_weights.dtype)
        .mean(dim=0)
        * F.one_hot(oracle, num_classes=gate_weights.shape[1])
        .to(dtype=gate_weights.dtype)
        .mean(dim=0)
    ).sum()
    teacher_entropy = -(
        teacher_weights * (teacher_weights + epsilon).log()
    ).sum(dim=-1).mean()
    return {
        "loss": router_loss,
        "top1_match": match,
        "teacher_top1_match": teacher_match,
        "expected_match": expected_match,
        "regret": regret,
        "teacher_entropy": teacher_entropy,
    }


def aggregate_group_loss(
    position_loss,
    root_group_size,
    mode="mean",
    cvar_fraction=0.25,
    cvar_weight=0.5,
):
    if root_group_size <= 0 or position_loss.numel() % root_group_size:
        raise ValueError("position loss does not align to root groups")
    if not 0.0 < cvar_fraction <= 1.0 or not 0.0 <= cvar_weight <= 1.0:
        raise ValueError("invalid CVaR fraction or weight")
    grouped = position_loss.reshape(-1, root_group_size)
    mean_loss = grouped.mean(dim=1)
    tail_count = max(1, int(math.ceil(root_group_size * cvar_fraction)))
    cvar_loss = grouped.topk(tail_count, dim=1).values.mean(dim=1)
    if mode == "mean":
        selected = mean_loss
    elif mode == "cvar":
        selected = cvar_loss
    elif mode == "mixed":
        selected = (1.0 - cvar_weight) * mean_loss + cvar_weight * cvar_loss
    else:
        raise ValueError(f"unknown group loss mode: {mode}")
    return selected.mean(), mean_loss.mean(), cvar_loss.mean()


class ExpertBlendingLightningModule(pl.LightningModule):
    """Expert Blending モデルの学習モジュール。

    損失関数は既存 NNUE と同じ:
      loss = λ * teacher_loss + (1 - λ) * outcome_loss - entropy
    """

    def __init__(
        self,
        model,
        lr_nnue=0.5,
        lr_adapter=0.5,
        lambda_=1.0,
        label_smoothing_eps=0.0,
        score_scaling=361,
        num_batches_warmup=10000,
        newbob_decay=0.5,
        num_epochs_to_adjust_lr=50,
        min_newbob_scale=1e-5,
        momentum=0.0,
        lambda_sparse=0.0,
        lambda_balance=0.0,
        gate_transform="softmax",
        router_teacher_mode="none",
        router_teacher_temperature=0.005,
        lambda_router=0.0,
        task_loss_weight=1.0,
        root_group_size=1,
        group_loss_mode="mean",
        group_cvar_fraction=0.25,
        group_cvar_weight=0.5,
        root_feature_dim=0,
        lambda_role_expert=0.0,
    ):
        super().__init__()
        self.model = model
        self.lr_nnue = lr_nnue
        self.lr_adapter = lr_adapter
        self.lambda_ = lambda_
        self.label_smoothing_eps = label_smoothing_eps
        self.score_scaling = score_scaling
        self.num_batches_warmup = num_batches_warmup
        self.newbob_decay = newbob_decay
        self.num_epochs_to_adjust_lr = num_epochs_to_adjust_lr
        self.min_newbob_scale = min_newbob_scale
        self.momentum = momentum
        self.lambda_sparse = lambda_sparse
        self.lambda_balance = lambda_balance
        self.gate_transform = gate_transform
        self.router_teacher_mode = router_teacher_mode
        self.router_teacher_temperature = router_teacher_temperature
        self.lambda_router = lambda_router
        self.task_loss_weight = task_loss_weight
        self.root_group_size = int(root_group_size)
        self.group_loss_mode = group_loss_mode
        self.group_cvar_fraction = float(group_cvar_fraction)
        self.group_cvar_weight = float(group_cvar_weight)
        self.root_feature_dim = int(root_feature_dim)
        self.lambda_role_expert = float(lambda_role_expert)
        if self.root_group_size <= 0:
            raise ValueError("root_group_size must be positive")

        # NewBob state
        self.newbob_scale = 1.0
        self.best_loss = 1e10
        self.warmup_start_global_step = 0
        self.latest_loss_sum = 0.0
        self.latest_loss_count = 0

        self.save_hyperparameters(ignore=["model"])
        self.backbone_type = getattr(model, "backbone_type", "dnn")

    def forward(self, *inputs, training=True, return_gate=False):
        return self.model(
            *inputs, training=training, return_gate=return_gate
        )

    def _compute_loss(self, batch, loss_type):
        base_batch_length = 11 if self.backbone_type == "nnue" else 9
        expert_roles = None
        if self.lambda_role_expert:
            expert_roles = batch[-1]
            batch = batch[:-1]
        extra_index = base_batch_length
        root_auxiliary = None
        if self.root_feature_dim:
            if len(batch) <= extra_index:
                raise RuntimeError("root auxiliary features are missing from the batch")
            root_auxiliary = batch[extra_index]
            extra_index += 1
        teacher_losses = batch[extra_index] if len(batch) > extra_index else None
        batch = batch[:base_batch_length]
        if self.backbone_type == "nnue":
            us_bb, them_bb, white_bb, black_bb, us, them, white, black, outcome, score, ply = batch
            model_inputs = (us_bb, them_bb, white_bb, black_bb, us, them, white, black)
            batch_size = int(us_bb.shape[0])
        else:
            x1, x2, us, them, white, black, outcome, score, ply = batch
            model_inputs = (x1, x2, us, them, white, black)
            batch_size = int(x1.shape[0])

        nnue2score = 600
        scaling = self.score_scaling

        if self.root_group_size > 1:
            if self.backbone_type != "dnn":
                raise RuntimeError("root-grouped training currently requires DNN backbone")
            x1, x2, us, them, white, black = model_inputs
            root_features = self.model.backbone(x1, x2)
            gate_weights = self.model.adapter(
                root_features, auxiliary=root_auxiliary, training=self.training
            )
            expanded_gate = gate_weights.repeat_interleave(
                self.root_group_size, dim=0
            )
            if int(us.shape[0]) != batch_size * self.root_group_size:
                raise RuntimeError("leaf batch does not match root_group_size")
            raw_value = self.model.nnue_experts(
                expanded_gate, us, them, white, black
            )
            role_raw_value = None
            if expert_roles is not None:
                if expert_roles.shape != (batch_size,):
                    raise ValueError("expert roles must have one index per root")
                if int(expert_roles.min()) < 0 or int(expert_roles.max()) >= gate_weights.shape[1]:
                    raise ValueError("expert role index is out of range")
                role_gate = F.one_hot(
                    expert_roles, num_classes=gate_weights.shape[1]
                ).to(dtype=gate_weights.dtype)
                role_raw_value = self.model.nnue_experts(
                    role_gate.repeat_interleave(self.root_group_size, dim=0),
                    us, them, white, black,
                )
        else:
            raw_value, gate_weights = self(
                *model_inputs, training=self.training, return_gate=True
            )
        q = raw_value * nnue2score / scaling
        t = outcome * (1.0 - self.label_smoothing_eps * 2.0) + self.label_smoothing_eps
        p = (score / scaling).sigmoid()

        epsilon = 1e-12
        teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
        outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
        teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
        outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))

        lambda_ = self.lambda_
        result = lambda_ * teacher_loss + (1.0 - lambda_) * outcome_loss
        entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy
        position_loss = result - entropy
        role_expert_loss = position_loss.new_zeros(())
        if expert_roles is not None:
            role_q = role_raw_value * nnue2score / scaling
            role_teacher_loss = -(
                p * F.logsigmoid(role_q) + (1.0 - p) * F.logsigmoid(-role_q)
            )
            role_outcome_loss = -(
                t * F.logsigmoid(role_q) + (1.0 - t) * F.logsigmoid(-role_q)
            )
            role_position_loss = (
                lambda_ * role_teacher_loss
                + (1.0 - lambda_) * role_outcome_loss
                - entropy
            )
            role_expert_loss, _, _ = aggregate_group_loss(
                role_position_loss, self.root_group_size, mode="mean"
            )
        mean_group_loss = position_loss.mean()
        cvar_group_loss = position_loss.mean()
        if self.root_group_size > 1:
            task_loss, mean_group_loss, cvar_group_loss = aggregate_group_loss(
                position_loss,
                self.root_group_size,
                self.group_loss_mode,
                self.group_cvar_fraction,
                self.group_cvar_weight,
            )
        else:
            if self.group_loss_mode != "mean":
                raise RuntimeError("tail group objectives require root-grouped data")
            task_loss = position_loss.mean()
        gate_statistics = compute_gate_statistics(gate_weights)
        gate_entropy = gate_statistics["entropy"]
        balance_kl = gate_statistics["balance_kl"]
        regularization = (
            self.lambda_sparse * gate_entropy
            + self.lambda_balance * balance_kl
        )
        router_statistics = None
        router_loss = task_loss.new_zeros(())
        if teacher_losses is not None:
            n_experts = gate_weights.shape[1]
            if self.router_teacher_mode == "cached":
                if teacher_losses.shape[1] != 2 * n_experts:
                    raise ValueError(
                        "cached gate teachers require [expert losses, weights]"
                    )
                teacher_weights = teacher_losses[:, n_experts:].detach()
                teacher_losses = teacher_losses[:, :n_experts]
            else:
                metric_mode = (
                    self.router_teacher_mode
                    if self.router_teacher_mode != "none"
                    else "hard"
                )
                teacher_weights = router_teacher_distribution(
                    teacher_losses.detach(),
                    metric_mode,
                    self.router_teacher_temperature,
                )
            router_statistics = compute_router_statistics(
                gate_weights, teacher_losses.detach(), teacher_weights
            )
            router_loss = router_statistics["loss"]
        elif self.router_teacher_mode != "none":
            raise RuntimeError("router distillation requires a teacher cache")
        loss = (
            self.task_loss_weight * task_loss
            + self.lambda_router * router_loss
            + self.lambda_role_expert * role_expert_loss
            + regularization
        )
        effective_experts = gate_statistics["effective_experts"]
        max_weight = gate_statistics["max_weight"]
        top2_mass = gate_statistics["top2_mass"]
        if loss_type == "train_loss":
            # Step-wise trace for debugging/volatility checks.
            self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=False)
            # Epoch aggregate to compare against val_loss (same granularity).
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log("train_task_loss", task_loss, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch_size)
            if expert_roles is not None:
                self.log("train_role_expert_loss", role_expert_loss, on_step=False,
                         on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("train_gate_entropy", gate_entropy, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch_size)
            self.log("train_balance_kl", balance_kl, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch_size)
            if router_statistics is not None:
                self._log_router_statistics(
                    "train", router_statistics, batch_size
                )
        else:
            self.log("val_loss", task_loss, on_step=False, on_epoch=True,
                     prog_bar=True, batch_size=batch_size)
            self.log("val_mean_group_loss", mean_group_loss, on_step=False,
                     on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("val_cvar_group_loss", cvar_group_loss, on_step=False,
                     on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("val_regularized_loss", loss, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch_size)
            self.log("val_gate_entropy", gate_entropy, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch_size)
            self.log("val_effective_experts", effective_experts, on_step=False,
                     on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("val_gate_max_weight", max_weight, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch_size)
            self.log("val_gate_top2_mass", top2_mass, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch_size)
            self.log("val_balance_kl", balance_kl, on_step=False, on_epoch=True,
                     prog_bar=False, batch_size=batch_size)
            if expert_roles is not None:
                self.log("val_role_expert_loss", role_expert_loss, on_step=False,
                         on_epoch=True, prog_bar=False, batch_size=batch_size)
            if router_statistics is not None:
                self._log_router_statistics("val", router_statistics, batch_size)
        return loss

    def _log_router_statistics(self, prefix, statistics, batch_size):
        for name, value in statistics.items():
            self.log(
                f"{prefix}_router_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "train_loss")

        # Expert 重み分布のログ (100 step ごと)
        if self.global_step % 100 == 0:
            self._log_expert_weights(batch)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val_loss")

    def validation_epoch_end(self, outputs):
        self.latest_loss_sum += float(sum(outputs)) / len(outputs)
        self.latest_loss_count += 1

        if (
            self.newbob_decay != 1.0
            and self.current_epoch > 0
            and self.current_epoch % self.num_epochs_to_adjust_lr == 0
        ):
            latest_loss = self.latest_loss_sum / self.latest_loss_count
            self.latest_loss_sum = 0.0
            self.latest_loss_count = 0
            if latest_loss < self.best_loss:
                self.print(
                    f"{self.current_epoch=}, {latest_loss=} < {self.best_loss=}, "
                    f"accepted, {self.newbob_scale=}"
                )
                sys.stdout.flush()
                self.best_loss = latest_loss
            else:
                self.newbob_scale *= self.newbob_decay
                self.print(
                    f"{self.current_epoch=}, {latest_loss=} >= {self.best_loss=}, "
                    f"rejected, {self.newbob_scale=}"
                )
                sys.stdout.flush()

        if self.newbob_scale < self.min_newbob_scale:
            self.trainer.should_stop = True
            self.print(f"{self.current_epoch=}, early stopping")

    def _log_expert_weights(self, batch):
        """バッチ内の expert 重み分布をログに記録する。"""
        with torch.no_grad():
            if self.backbone_type == "nnue":
                us_bb, them_bb, white_bb, black_bb = batch[0], batch[1], batch[2], batch[3]
                gate_weights = self.model.backbone(
                    us_bb, them_bb, white_bb, black_bb, training=False
                )
            else:
                x1, x2 = batch[0], batch[1]
                feat = self.model.backbone(x1, x2)
                auxiliary = batch[9] if self.root_feature_dim else None
                gate_weights = self.model.adapter(
                    feat, auxiliary=auxiliary, training=False
                )
            # 各 expert の平均重み
            mean_weights = gate_weights.mean(dim=0)
            for i in range(mean_weights.shape[0]):
                self.log(f"expert_weight/expert_{i}", mean_weights[i])
            # expert 重みのエントロピー (均等度の指標)
            entropy = -(gate_weights * (gate_weights + 1e-12).log()).sum(dim=-1).mean()
            self.log("expert_weight/entropy", entropy)

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx,
        optimizer_closure, on_tpu, using_native_amp, using_lbfgs,
    ):
        # Linear warmup
        if self.trainer.global_step - self.warmup_start_global_step < self.num_batches_warmup:
            warmup_scale = min(
                1.0,
                float(self.trainer.global_step - self.warmup_start_global_step + 1)
                / self.num_batches_warmup,
            )
        else:
            warmup_scale = 1.0

        for pg in optimizer.param_groups:
            base_lr = pg["initial_lr"]
            pg["lr"] = base_lr * warmup_scale * self.newbob_scale
        self.log("lr", optimizer.param_groups[0]["lr"])

        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        # Separate param groups: adapter と NNUE experts で異なる学習率
        param_groups = []
        if self.backbone_type == "nnue":
            param_groups.append(
                {
                    "params": list(self.model.backbone.parameters()),
                    "lr": self.lr_adapter,
                    "initial_lr": self.lr_adapter,
                }
            )
        else:
            param_groups.append(
                {
                    "params": list(self.model.adapter.parameters()),
                    "lr": self.lr_adapter,
                    "initial_lr": self.lr_adapter,
                }
            )
        expert_parameters = [
            p for p in self.model.nnue_experts.parameters() if p.requires_grad
        ]
        if expert_parameters:
            param_groups.append(
                {
                    "params": expert_parameters,
                    "lr": self.lr_nnue,
                    "initial_lr": self.lr_nnue,
                }
            )
        return torch.optim.SGD(param_groups, lr=self.lr_nnue, momentum=self.momentum)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["custom_state"] = {
            "newbob_scale": self.newbob_scale,
            "best_loss": self.best_loss,
            "warmup_start_global_step": self.warmup_start_global_step,
            "latest_loss_sum": self.latest_loss_sum,
            "latest_loss_count": self.latest_loss_count,
        }

    def on_load_checkpoint(self, checkpoint):
        if "custom_state" in checkpoint:
            state = checkpoint["custom_state"]
            self.newbob_scale = state["newbob_scale"]
            self.best_loss = state["best_loss"]
            self.warmup_start_global_step = state["warmup_start_global_step"]
            self.latest_loss_sum = state["latest_loss_sum"]
            self.latest_loss_count = state["latest_loss_count"]


class CheckpointEveryNEpochs(pl.callbacks.Checkpoint):
    """指定エポック間隔でチェックポイントを保存するコールバック。"""

    def __init__(self, every_n_epochs, log_dir):
        self.every_n_epochs = every_n_epochs
        self.log_dir = log_dir

    def on_validation_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        ckpt_path = os.path.join(self.log_dir, f"{trainer.current_epoch}.ckpt")
        trainer.save_checkpoint(ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Expert Blending model training")
    # Data
    parser.add_argument(
        "--train",
        required=True,
        help="Training split directory (contains dnn.bin and nnue.bin)",
    )
    parser.add_argument(
        "--val",
        required=True,
        help="Validation split directory (contains dnn.bin and nnue.bin)",
    )
    parser.add_argument("--feature-set", default="HalfKP", help="NNUE feature set name")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epoch-size", type=int, default=1000000, help="Positions per epoch")
    parser.add_argument(
        "--root-grouped",
        action="store_true",
        help="Treat train/val as root-grouped directories; epoch sizes count roots",
    )
    # Model
    parser.add_argument(
        "--backbone-type",
        default="dnn",
        choices=["dnn", "nnue"],
        help="Backbone type: dnn or nnue",
    )
    parser.add_argument("--backbone-weights", required=False, help="dlshogi .npz weights path")
    parser.add_argument("--nnue-checkpoint", required=True, help="NNUE .ckpt path for expert init")
    parser.add_argument("--n-experts", type=int, default=4, help="Number of NNUE experts")
    parser.add_argument(
        "--blend-mode",
        default="weighted",
        choices=["weighted", "residual"],
        help="NNUE expert blending mode",
    )
    parser.add_argument("--adapter-hidden", type=int, default=128, help="Adapter hidden dim")
    parser.add_argument("--root-feature-dim", type=int, default=0)
    parser.add_argument("--train-root-features")
    parser.add_argument("--val-root-features")
    parser.add_argument("--train-expert-roles")
    parser.add_argument("--val-expert-roles")
    parser.add_argument(
        "--adapter-noise-scale",
        type=float,
        default=1.0,
        help="Gaussian noise scale added to adapter logits during training",
    )
    parser.add_argument(
        "--gate-transform",
        default="softmax",
        choices=["softmax", "entmax15"],
        help="Probability transform applied to gate logits",
    )
    # Training
    parser.add_argument("--lr-nnue", type=float, default=0.5, help="LR for NNUE experts")
    parser.add_argument("--lr-adapter", type=float, default=0.5, help="LR for DNN adapter")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_",
                        help="1.0=teacher scores, 0.0=game results")
    parser.add_argument("--label-smoothing-eps", type=float, default=0.0)
    parser.add_argument("--score-scaling", type=float, default=361)
    parser.add_argument("--num-batches-warmup", type=int, default=10000)
    parser.add_argument("--newbob-decay", type=float, default=0.5)
    parser.add_argument("--num-epochs-to-adjust-lr", type=int, default=50)
    parser.add_argument("--min-newbob-scale", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument(
        "--lambda-sparse",
        type=float,
        default=0.0,
        help="Coefficient for mean per-position gate entropy",
    )
    parser.add_argument(
        "--lambda-balance",
        type=float,
        default=0.0,
        help="Coefficient for KL(mean gate weights || uniform)",
    )
    parser.add_argument("--train-teacher-cache")
    parser.add_argument("--val-teacher-cache")
    parser.add_argument(
        "--router-teacher-mode",
        default="none",
        choices=["none", "hard", "soft", "cached"],
    )
    parser.add_argument("--router-teacher-temperature", type=float, default=0.005)
    parser.add_argument("--lambda-router", type=float, default=0.0)
    parser.add_argument("--task-loss-weight", type=float, default=1.0)
    parser.add_argument("--lambda-role-expert", type=float, default=0.0)
    parser.add_argument(
        "--group-loss-mode",
        choices=["mean", "cvar", "mixed"],
        default="mean",
    )
    parser.add_argument("--group-cvar-fraction", type=float, default=0.25)
    parser.add_argument("--group-cvar-weight", type=float, default=0.5)
    parser.add_argument("--freeze-experts", action="store_true")
    parser.add_argument("--freeze-adapter", action="store_true")
    parser.add_argument("--max-val-positions", type=int, default=100000,
                        help="Max validation positions per epoch")
    parser.add_argument("--val-start-position", type=int, default=0)
    parser.add_argument(
        "--train-shuffle-buffer-size",
        type=int,
        default=0,
        help="Batch-level shuffle buffer size for train loader (0 disables)",
    )
    parser.add_argument("--network-save-period", type=int, default=100,
                        help="Epochs between checkpoint saves")
    # PyTorch Lightning
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--default-root-dir", default="logs/expert_blending_v1")
    parser.add_argument("--seed", type=int, default=42)
    # Resume
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Resume full training state from .ckpt")
    parser.add_argument("--load-weights-only", default=None,
                        help="Load model weights only from .ckpt (no optimizer/lr/epoch state). "
                             "Ignored if --resume-from-checkpoint is also given.")

    args = parser.parse_args()

    if (
        args.lambda_sparse < 0.0
        or args.lambda_balance < 0.0
        or args.lambda_router < 0.0
        or args.task_loss_weight < 0.0
        or args.lambda_role_expert < 0.0
        or not 0.0 < args.group_cvar_fraction <= 1.0
        or not 0.0 <= args.group_cvar_weight <= 1.0
    ):
        parser.error("loss coefficients must be non-negative")
    if args.router_teacher_temperature <= 0.0:
        parser.error("--router-teacher-temperature must be positive")
    if args.router_teacher_mode != "none" and (
        not args.train_teacher_cache or not args.val_teacher_cache
    ):
        parser.error(
            "router distillation requires both train and validation teacher caches"
        )
    if args.root_feature_dim < 0:
        parser.error("--root-feature-dim must be non-negative")
    if args.root_feature_dim and (
        not args.root_grouped
        or not args.train_root_features
        or not args.val_root_features
    ):
        parser.error("root features require root-grouped data and both feature caches")
    if not args.root_feature_dim and (
        args.train_root_features or args.val_root_features
    ):
        parser.error("feature cache paths require --root-feature-dim")
    if args.lambda_role_expert and (
        not args.root_grouped
        or not args.train_expert_roles
        or not args.val_expert_roles
    ):
        parser.error("role expert loss requires root-grouped data and both role caches")

    root_group_size = 1
    if args.root_grouped:
        if args.backbone_type != "dnn":
            parser.error("--root-grouped currently supports only --backbone-type dnn")
        import json

        group_sizes = []
        for directory in (args.train, args.val):
            with open(os.path.join(directory, "metadata.json")) as file:
                group_sizes.append(int(json.load(file)["group_size"]))
        if group_sizes[0] != group_sizes[1]:
            parser.error("root-grouped train and validation group sizes differ")
        root_group_size = group_sizes[0]

    required_paths = [args.train, args.val, args.nnue_checkpoint]
    if args.backbone_type == "dnn":
        if not args.backbone_weights:
            raise ValueError("--backbone-weights is required when --backbone-type dnn")
        required_paths.append(args.backbone_weights)
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")

    pl.seed_everything(args.seed)

    feature_set = nnue_features.get_feature_set_from_name(args.feature_set)
    print(f"Feature set: {feature_set.name}")
    print(f"Num features: {feature_set.num_features}")

    # --- Device ---
    main_device = "cuda:0" if args.gpus > 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {main_device}")

    # --- Model ---
    print("Building Expert Blending model...")
    model = create_expert_blending_model(
        backbone_weights_path=args.backbone_weights,
        nnue_ckpt_path=args.nnue_checkpoint,
        feature_set=feature_set,
        n_experts=args.n_experts,
        adapter_hidden=args.adapter_hidden,
        adapter_auxiliary_dim=args.root_feature_dim,
        adapter_noise_scale=args.adapter_noise_scale,
        backbone_type=args.backbone_type,
        blend_mode=args.blend_mode,
        gate_transform=args.gate_transform,
        device="cpu",  # PL will move to GPU
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # --- Lightning module ---
    lit_module = ExpertBlendingLightningModule(
        model=model,
        lr_nnue=args.lr_nnue,
        lr_adapter=args.lr_adapter,
        lambda_=args.lambda_,
        label_smoothing_eps=args.label_smoothing_eps,
        score_scaling=args.score_scaling,
        num_batches_warmup=args.num_batches_warmup,
        newbob_decay=args.newbob_decay,
        num_epochs_to_adjust_lr=args.num_epochs_to_adjust_lr,
        min_newbob_scale=args.min_newbob_scale,
        momentum=args.momentum,
        lambda_sparse=args.lambda_sparse,
        lambda_balance=args.lambda_balance,
        gate_transform=args.gate_transform,
        router_teacher_mode=args.router_teacher_mode,
        router_teacher_temperature=args.router_teacher_temperature,
        lambda_router=args.lambda_router,
        task_loss_weight=args.task_loss_weight,
        root_group_size=root_group_size,
        group_loss_mode=args.group_loss_mode,
        group_cvar_fraction=args.group_cvar_fraction,
        group_cvar_weight=args.group_cvar_weight,
        root_feature_dim=args.root_feature_dim,
        lambda_role_expert=args.lambda_role_expert,
    )

    # --- Load weights only (for fine-tuning) ---
    if args.load_weights_only and not args.resume_from_checkpoint:
        print(f"Loading model weights only from: {args.load_weights_only}")
        ckpt = torch.load(args.load_weights_only, map_location="cpu")
        lit_module.load_state_dict(ckpt["state_dict"], strict=True)
        del ckpt
        print("Model weights loaded (optimizer/lr/epoch state NOT restored).")

    if args.freeze_experts:
        model.nnue_experts.requires_grad_(False)
        print("NNUE experts frozen; training router parameters only.")
    if args.freeze_adapter:
        if model.adapter is None:
            parser.error("--freeze-adapter requires DNN adapter")
        model.adapter.requires_grad_(False)
        print("DNN adapter frozen; training expert parameters only.")

    # --- Data ---
    print(f"Training: {args.train}")
    print(f"Validation: {args.val}")
    print(f"Batch size: {args.batch_size}, Epoch size: {args.epoch_size}")

    if args.root_grouped:
        train_loader, val_loader, loaded_group_size = create_root_grouped_loaders(
            train_directory=args.train,
            val_directory=args.val,
            feature_set_name=args.feature_set,
            root_batch_size=args.batch_size,
            device=main_device,
            epoch_roots=args.epoch_size,
            max_val_roots=args.max_val_positions,
            train_shuffle_buffer_size=args.train_shuffle_buffer_size,
            seed=args.seed,
            train_teacher_cache=args.train_teacher_cache,
            val_teacher_cache=args.val_teacher_cache,
            train_root_features=args.train_root_features,
            val_root_features=args.val_root_features,
            train_expert_roles=args.train_expert_roles,
            val_expert_roles=args.val_expert_roles,
        )
        if loaded_group_size != root_group_size:
            raise RuntimeError("root group size changed while constructing loaders")
    else:
        train_loader, val_loader = create_data_loaders(
            train_bin_dir=args.train,
            val_bin_dir=args.val,
            feature_set_name=args.feature_set,
            batch_size=args.batch_size,
            device=main_device,
            epoch_size=args.epoch_size,
            max_val_positions=args.max_val_positions,
            train_shuffle_buffer_size=args.train_shuffle_buffer_size,
            seed=args.seed,
            backbone_type=args.backbone_type,
            train_teacher_cache=args.train_teacher_cache,
            val_teacher_cache=args.val_teacher_cache,
            val_start_position=args.val_start_position,
        )

    # --- Trainer ---
    logdir = args.default_root_dir
    os.makedirs(logdir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(logdir)

    ckpt_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_callback = CheckpointEveryNEpochs(
        every_n_epochs=args.network_save_period, log_dir=ckpt_dir,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus if torch.cuda.is_available() else 0,
        logger=tb_logger,
        callbacks=[ckpt_callback],
        log_every_n_steps=50,
    )

    print(f"Log dir: {logdir}")
    print("Starting training...", flush=True)

    trainer.fit(
        lit_module, train_loader, val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )

    # Save final checkpoint
    final_path = os.path.join(tb_logger.log_dir, "final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"Final checkpoint saved: {final_path}")


if __name__ == "__main__":
    main()
