"""
Training script for EfficientSegMamba on 3D Brain Tumor Segmentation (BraTS 2023).
Supports:
- Pseudo-3D (P3D) Convolutions
- Shifted Softplus KAN (SKAN)
- Morton Z-order tokenization
- Adaptive Onsampling Decoder
- Dice + Focal Tversky Loss (as described in the manuscript)
- Automatic Mixed Precision (AMP)
"""

import os
import sys
import argparse
import tempfile
import warnings
from datetime import datetime

os.environ["TMPDIR"] = "/tmp"
tempfile.tempdir = "/tmp"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.utils import set_determinism

import configs.settings as settings
from light_training.dataloading.dataset import get_train_val_test_loader_from_split_json
from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last
from model_segmamba.segmamba import SegMamba

# Set global reproducibility
settings.set_global_reproducibility(settings.REPRO_SEED)
set_determinism(settings.REPRO_SEED)


class DiceFocalTverskyLoss(nn.Module):
    """
    Hybrid Dice + Focal Tversky Loss.
    Penalizes False Negatives more heavily via beta=0.7 to address severe
    class imbalance in small tumor sub-regions (ET / TC).
    """
    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        gamma=4.0 / 3.0,
        smooth=1e-6,
        lambda_dice=1.0,
        lambda_focal_tversky=1.0,
        include_background=False,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.lambda_dice = lambda_dice
        self.lambda_focal_tversky = lambda_focal_tversky
        self.include_background = include_background

    def forward(self, logits, target):
        if target.ndim == logits.ndim:
            target = target[:, 0]
        target = target.long()

        probs = torch.softmax(logits.float(), dim=1)
        target_onehot = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1.0)

        if not self.include_background and probs.shape[1] > 1:
            probs = probs[:, 1:]
            target_onehot = target_onehot[:, 1:]

        reduce_dims = tuple(range(2, probs.ndim))
        true_pos = (probs * target_onehot).sum(dim=reduce_dims)
        false_pos = (probs * (1.0 - target_onehot)).sum(dim=reduce_dims)
        false_neg = ((1.0 - probs) * target_onehot).sum(dim=reduce_dims)

        dice_score = (2.0 * true_pos + self.smooth) / (
            2.0 * true_pos + false_pos + false_neg + self.smooth
        )
        tversky_score = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )

        dice_loss = 1.0 - dice_score.mean()
        focal_term = (1.0 - tversky_score).clamp(min=self.smooth, max=1.0)
        focal_tversky_loss = torch.pow(focal_term, 1.0 / self.gamma).mean()
        return self.lambda_dice * dice_loss + self.lambda_focal_tversky * focal_tversky_loss


class EfficientSegMambaTrainer(Trainer):
    def __init__(
        self,
        env_type,
        max_epochs,
        batch_size,
        device="cpu",
        val_every=1,
        num_gpus=1,
        logdir="./logs/",
        master_ip="localhost",
        master_port=17750,
        training_script="train.py",
        patch_size=(96, 96, 96),
        amp_enabled=True,
    ):
        super().__init__(
            env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script
        )
        self.patch_size = patch_size
        self.window_infer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=4,
            overlap=0.5,
        )
        self.model = SegMamba()
        self.best_mean_dice = 0.0
        self.best_mean_hd95 = float("inf")
        self.train_process = settings.SEGMAMBA_TRAIN_PROCESS
        self.val_process = settings.SEGMAMBA_VAL_PROCESS

        self.optimizer = settings.build_optimizer(self.model)
        self.scheduler_builder = settings.build_scheduler

        # Loss function
        if settings.SEGMAMBA_LOSS_NAME == "DiceFocalTversky":
            self.loss_fn = DiceFocalTverskyLoss(
                alpha=settings.SEGMAMBA_FOCAL_TVERSKY_ALPHA,
                beta=settings.SEGMAMBA_FOCAL_TVERSKY_BETA,
                gamma=settings.SEGMAMBA_FOCAL_TVERSKY_GAMMA,
                lambda_dice=settings.SEGMAMBA_DICE_LOSS_WEIGHT,
                lambda_focal_tversky=settings.SEGMAMBA_FOCAL_TVERSKY_LOSS_WEIGHT,
                include_background=settings.SEGMAMBA_LOSS_INCLUDE_BACKGROUND,
            )
        else:
            self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

        self.amp_enabled = amp_enabled and torch.cuda.is_available()
        self.grad_scaler = torch.amp.GradScaler("cuda") if self.amp_enabled else None
        print(f"[*] EfficientSegMamba Model Instantiated.")
        print(f"[*] Loss: {settings.SEGMAMBA_LOSS_NAME} (alpha={settings.SEGMAMBA_FOCAL_TVERSKY_ALPHA}, beta={settings.SEGMAMBA_FOCAL_TVERSKY_BETA}, gamma={settings.SEGMAMBA_FOCAL_TVERSKY_GAMMA})")
        print(f"[*] AMP Enabled: {self.amp_enabled}")

    def training_step(self, batch):
        image, label = batch["data"], batch["seg"]
        image = image.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)

        if self.amp_enabled:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = self.model(image)
                loss = self.loss_fn(pred, label)
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()
        else:
            pred = self.model(image)
            loss = self.loss_fn(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=12.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()

    def validation_step(self, batch):
        image, label = batch["data"], batch["seg"]
        image = image.to(self.device)
        label = label.to(self.device)

        with torch.no_grad():
            if self.amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pred = self.window_infer(image, self.model)
            else:
                pred = self.window_infer(image, self.model)

        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1, keepdim=True)

        # BraTS Regions: TC, WT, ET
        def to_brats_regions(x):
            tc = (x == 1) | (x == 3)
            wt = (x == 1) | (x == 2) | (x == 3)
            et = (x == 3)
            return torch.cat([tc, wt, et], dim=1).float()

        pred_regions = to_brats_regions(pred)
        label_regions = to_brats_regions(label)

        dice_tc = dice(pred_regions[:, 0], label_regions[:, 0]).item()
        dice_wt = dice(pred_regions[:, 1], label_regions[:, 1]).item()
        dice_et = dice(pred_regions[:, 2], label_regions[:, 2]).item()
        mean_dice = (dice_tc + dice_wt + dice_et) / 3.0

        return {"dice_tc": dice_tc, "dice_wt": dice_wt, "dice_et": dice_et, "mean_dice": mean_dice}

    def validation_end(self, val_outputs):
        mean_dice = np.mean([x["mean_dice"] for x in val_outputs])
        mean_tc = np.mean([x["dice_tc"] for x in val_outputs])
        mean_wt = np.mean([x["dice_wt"] for x in val_outputs])
        mean_et = np.mean([x["dice_et"] for x in val_outputs])

        print(f"Validation Epoch {self.epoch}: Mean Dice: {mean_dice:.4f} (WT: {mean_wt:.4f}, TC: {mean_tc:.4f}, ET: {mean_et:.4f})")
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_path = os.path.join(self.logdir, "checkpoints", f"best_model_dice_{mean_dice:.4f}.pt")
            save_new_model_and_delete_last(self.model, save_path, delete_symbol="best_model_dice")
            print(f"[*] New Best Model Saved with Mean Dice: {mean_dice:.4f} -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train EfficientSegMamba on BraTS 2023")
    parser.add_argument("--data_dir", type=str, default="./data/fullres/train", help="Path to preprocessed train data")
    parser.add_argument("--split_json", type=str, default="./splits/brats23_split_70_10_20.json", help="Path to split JSON")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[96, 96, 96], help="3D input patch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--logdir", type=str, default="./logs/efficient_segmamba", help="Directory for logs and checkpoints")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision")
    args = parser.parse_args()

    settings.LEARNING_RATE = args.lr
    settings.PATCH_SIZE = args.patch_size

    trainer = EfficientSegMambaTrainer(
        env_type="pytorch",
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        val_every=args.val_every,
        num_gpus=args.num_gpus,
        logdir=args.logdir,
        patch_size=tuple(args.patch_size),
        amp_enabled=args.amp,
    )

    train_loader, val_loader, _ = get_train_val_test_loader_from_split_json(
        data_dir=args.data_dir,
        split_json_file=args.split_json,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        fold=0,
    )

    trainer.train(train_dataset=train_loader, val_dataset=val_loader)


if __name__ == "__main__":
    main()
