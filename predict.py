"""
Inference script for EfficientSegMamba.
Executes sliding window inference over 3D volumes and exports segmentation predictions.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from light_training.dataloading.dataset import get_train_val_test_loader_from_split_json
from light_training.prediction import Predictor
from light_training.trainer import Trainer
from model_segmamba.segmamba import SegMamba
import configs.settings as settings

set_determinism(settings.REPRO_SEED)


class BraTSPredictor(Trainer):
    def __init__(
        self,
        checkpoint_path: str,
        patch_size=(96, 96, 96),
        overlap=0.5,
        device="cuda:0",
        save_path="./prediction_results/efficient_segmamba",
    ):
        super().__init__(
            env_type="pytorch",
            max_epochs=1,
            batch_size=1,
            device=device if torch.cuda.is_available() else "cpu",
            val_every=1,
            num_gpus=1,
            logdir="./logs/",
        )
        self.patch_size = patch_size
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.model = SegMamba()
        if os.path.exists(checkpoint_path):
            print(f"[*] Loading checkpoint: {checkpoint_path}")
            sd = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            new_sd = self.filte_state_dict(sd)
            self.model.load_state_dict(new_sd)
        else:
            print(f"[!] Warning: Checkpoint path does not exist: {checkpoint_path}. Running with uninitialized weights.")

        self.model.to(self.device)
        self.model.eval()

        self.window_infer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=2,
            overlap=overlap,
            progress=True,
            mode="gaussian",
        )
        self.predictor = Predictor(window_infer=self.window_infer, mirror_axes=[0, 1, 2])

    def convert_labels(self, labels):
        # WT, TC, ET regions
        result = [
            (labels == 1) | (labels == 3),
            (labels == 1) | (labels == 3) | (labels == 2),
            labels == 3,
        ]
        return torch.cat(result, dim=1).float()

    def predict_all(self, test_loader):
        print(f"[*] Running inference on {len(test_loader)} test cases...")
        for batch in test_loader:
            image = batch["data"]
            properties = batch.get("properties", {})
            case_id = properties.get("name", ["case_pred"])[0] if isinstance(properties, dict) else "pred"

            image = image.to(self.device)
            with torch.no_grad():
                pred = self.predictor.predict_single_case(self.model, image)

            save_file = os.path.join(self.save_path, f"{case_id}.npy")
            np.save(save_file, pred.cpu().numpy())
            print(f"Saved prediction: {save_file}")


def main():
    parser = argparse.ArgumentParser(description="Predict with EfficientSegMamba")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default="./data/fullres/train", help="Path to preprocessed MRI directory")
    parser.add_argument("--split_json", type=str, default="./splits/brats23_split_70_10_20.json", help="Path to split JSON file")
    parser.add_argument("--save_path", type=str, default="./prediction_results/efficient_segmamba", help="Output directory")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[96, 96, 96], help="Inference ROI size")
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap fraction")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on (e.g., cuda:0 or cpu)")
    args = parser.parse_args()

    _, _, test_loader = get_train_val_test_loader_from_split_json(
        data_dir=args.data_dir,
        split_json_file=args.split_json,
        batch_size=1,
        patch_size=args.patch_size,
        fold=0,
    )

    predictor = BraTSPredictor(
        checkpoint_path=args.checkpoint,
        patch_size=tuple(args.patch_size),
        overlap=args.overlap,
        device=args.device,
        save_path=args.save_path,
    )
    predictor.predict_all(test_loader)


if __name__ == "__main__":
    main()
