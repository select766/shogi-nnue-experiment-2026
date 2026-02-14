"""
Expert Blending モデルの学習スクリプト。

PyTorch Lightning ベースで、既存 NNUE と同じ損失関数
(teacher_loss + outcome_loss の λ ブレンド) を使用する。
勾配は DNN_adapter と NNUE_weights にのみ流す (backbone は frozen)。

Usage:
    cd nnue-pytorch && source .venv/bin/activate
    PYTHONPATH=../src:$PYTHONPATH python -m train_nnue.train_expert_blending \
        --train ../dataset_qsearch_split/train.bin \
        --val ../dataset_qsearch_split/val.bin \
        --backbone-weights ../tmp/dlshogi-model/model_resnet10_swish-072 \
        --nnue-checkpoint logs/halfkp_v1/checkpoints/83000.ckpt
"""

import argparse
import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

import features as nnue_features
from train_nnue.expert_blending_dataset import (
    ExpertBlendingDataset,
    FixedNumBatchesDataset,
    RECORD_BYTES,
)
from train_nnue.expert_blending_model import create_expert_blending_model


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

        # NewBob state
        self.newbob_scale = 1.0
        self.best_loss = 1e10
        self.warmup_start_global_step = 0
        self.latest_loss_sum = 0.0
        self.latest_loss_count = 0

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x1, x2, us, them, w_in, b_in, training=True):
        return self.model(x1, x2, us, them, w_in, b_in, training=training)

    def _compute_loss(self, batch, loss_type):
        x1, x2, us, them, white, black, outcome, score, ply = batch

        nnue2score = 600
        scaling = self.score_scaling

        q = self(x1, x2, us, them, white, black, training=self.training) * nnue2score / scaling
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
        loss = result.mean() - entropy.mean()

        self.log(loss_type, loss, prog_bar=True)
        return loss

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
        x1, x2 = batch[0], batch[1]
        with torch.no_grad():
            feat = self.model.backbone(x1, x2)
            gate_weights = self.model.adapter(feat, training=False)
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
        param_groups = [
            {
                "params": list(self.model.adapter.parameters()),
                "lr": self.lr_adapter,
                "initial_lr": self.lr_adapter,
            },
            {
                "params": list(self.model.nnue_experts.parameters()),
                "lr": self.lr_nnue,
                "initial_lr": self.lr_nnue,
            },
        ]
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
        if trainer.current_epoch == 0 or trainer.current_epoch % self.every_n_epochs != 0:
            return
        ckpt_path = os.path.join(self.log_dir, f"{trainer.current_epoch}.ckpt")
        trainer.save_checkpoint(ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Expert Blending model training")
    # Data
    parser.add_argument("--train", required=True, help="Training data (.bin)")
    parser.add_argument("--val", required=True, help="Validation data (.bin)")
    parser.add_argument("--feature-set", default="HalfKP", help="NNUE feature set name")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epoch-size", type=int, default=1000000, help="Positions per epoch")
    # Model
    parser.add_argument("--backbone-weights", required=True, help="dlshogi .npz weights path")
    parser.add_argument("--nnue-checkpoint", required=True, help="NNUE .ckpt path for expert init")
    parser.add_argument("--n-experts", type=int, default=4, help="Number of NNUE experts")
    parser.add_argument("--adapter-hidden", type=int, default=128, help="Adapter hidden dim")
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
    parser.add_argument("--max-val-positions", type=int, default=100000,
                        help="Max validation positions per epoch")
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

    args = parser.parse_args()

    for path in [args.train, args.val, args.backbone_weights, args.nnue_checkpoint]:
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
    )

    # --- Data ---
    print(f"Training: {args.train}")
    print(f"Validation: {args.val}")
    print(f"Batch size: {args.batch_size}, Epoch size: {args.epoch_size}")

    train_dataset = ExpertBlendingDataset(
        args.train, args.feature_set, args.batch_size, device=main_device, shuffle=True,
    )
    val_dataset = ExpertBlendingDataset(
        args.val, args.feature_set, args.batch_size, device=main_device, shuffle=False,
    )

    num_train_batches = (args.epoch_size + args.batch_size - 1) // args.batch_size
    val_records = os.path.getsize(args.val) // RECORD_BYTES
    num_val_batches = (min(val_records, args.max_val_positions) + args.batch_size - 1) // args.batch_size

    train_loader = DataLoader(
        FixedNumBatchesDataset(train_dataset, num_train_batches),
        batch_size=None, batch_sampler=None,
    )
    val_loader = DataLoader(
        FixedNumBatchesDataset(val_dataset, num_val_batches),
        batch_size=None, batch_sampler=None,
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
