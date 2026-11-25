#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-throughput PyTorch Dataset & DataLoader for preprocessed BraTS 2023 GLI (.npy + .pkl).

Reads directly from `data/` and the patient-level split JSON
`script/output_brats23_analysis/brats23_patient_split_70_10_20.json`.

Key features:
1. Two fast I/O backends (guided by our `io_benchmark.md`):
   - `cache_mode="mmap"`: Zero-copy `np.load(..., mmap_mode="r")` patch slicing (~13-17 ms/case,
     minimal RAM footprint across multi-worker DataLoaders; ideal with `/dev/shm` or NFS).
   - `cache_mode="ram"`: Preloads all split volumes into host RAM (~10 ms/case slice).
2. Foreground-oversampled 3D patch cropping (`patch_size=(128, 128, 128)`):
   Uses pre-computed `class_locations` in `<case_id>.pkl` (`label_1`, `label_2`, `label_3`, `wt`)
   to sample tumor-centered patches without calling `np.argwhere` during training.
3. Artifact-safe 3D augmentation (flips, 90-deg axial rotation, brain-masked intensity
   scale/shift/noise) applied strictly to nonzero brain foreground so background stays 0.0.
4. Supports both BraTS composite `regions` (`WT`, `TC`, `ET` -> 3 channels) and `multiclass`
   (`0, 1, 2, 3`) label outputs.

Run built-in sanity check:
    python dataset.py --sanity-check --cache-mode mmap --batch-size 4 --num-workers 2
"""

from __future__ import annotations

import os
os.environ["TMPDIR"] = "/tmp"
import tempfile
tempfile.tempdir = "/tmp"

import argparse
import json
import pickle
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
ALL_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = ALL_DIR.parent / "data"
DEFAULT_SPLIT_JSON = SCRIPT_DIR / "output_brats23_analysis" / "brats23_patient_split_70_10_20.json"


def parse_patient_id(case_id: str) -> str:
    """Extract patient ID from `BraTS-GLI-XXXXX-YYY`."""
    match = re.match(r"^BraTS-GLI-(\d+)-(\d+)$", case_id)
    if match:
        return match.group(1)
    parts = case_id.split("-")
    return parts[-2] if len(parts) >= 4 else case_id


def load_split_case_ids(
    split_json: Path,
    split_name: str,
    fold: int = 0,
) -> List[str]:
    """Load case IDs for `train`, `val`, or `test` from 70/10/20 or K-fold split JSON."""
    with split_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "folds" in data:
        if split_name not in ("train", "val"):
            raise ValueError(f"K-fold JSON only supports 'train' or 'val', got '{split_name}'.")
        fold_entry = data["folds"][fold]
        items = fold_entry[split_name]
    else:
        key = "val" if split_name == "val" and "val" in data else split_name
        if key not in data:
            raise KeyError(f"Split '{split_name}' not found in {split_json}. Keys: {list(data.keys())}")
        items = data[key]

    return [
        item["id"] if isinstance(item, dict) and "id" in item else Path(str(item)).stem
        for item in items
    ]


def encode_seg_target(seg: np.ndarray, label_mode: str) -> torch.Tensor:
    """Convert raw `{0,1,2,3}` mask into `regions` (3, X, Y, Z) or `multiclass` (X, Y, Z)."""
    if label_mode == "regions":
        wt = (seg > 0).astype(np.float32)
        tc = ((seg == 1) | (seg == 3)).astype(np.float32)
        et = (seg == 3).astype(np.float32)
        return torch.from_numpy(np.stack([wt, tc, et], axis=0))
    if label_mode == "multiclass":
        return torch.from_numpy(seg.astype(np.int64))
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def pad_to_min_shape(
    img: np.ndarray,
    seg: Optional[np.ndarray],
    target_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Symmetrically zero-pad `(C, X, Y, Z)` and `(X, Y, Z)` if smaller than `target_shape`."""
    cur_shape = img.shape[1:]
    need_pad = any(cur < tgt for cur, tgt in zip(cur_shape, target_shape))
    if not need_pad:
        return img, seg

    pad_spatial: List[Tuple[int, int]] = []
    for cur, tgt in zip(cur_shape, target_shape):
        if cur < tgt:
            diff = tgt - cur
            before = diff // 2
            after = diff - before
            pad_spatial.append((before, after))
        else:
            pad_spatial.append((0, 0))

    img_padded = np.pad(img, [(0, 0), *pad_spatial], mode="constant", constant_values=0.0)
    seg_padded = (
        np.pad(seg, pad_spatial, mode="constant", constant_values=0) if seg is not None else None
    )
    return img_padded, seg_padded


class BraTS23PreprocessedDataset(Dataset):
    """PyTorch Dataset for preprocessed BraTS 2023 `.npy` + `_seg.npy` + `.pkl` data."""

    def __init__(
        self,
        data_dir: Path | str,
        case_ids: Sequence[str],
        is_train: bool = True,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        fg_prob: float = 0.67,
        label_mode: str = "regions",
        cache_mode: str = "mmap",
        has_seg: bool = True,
        augment: Optional[bool] = None,
        aug_mode: str = "strong",
        modality_dropout_prob: float = 0.0,
        seed: int = 42,
        ram_preload_workers: int = 8,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.case_ids = list(case_ids)
        self.is_train = bool(is_train)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.fg_prob = float(fg_prob)
        self.label_mode = label_mode
        self.cache_mode = cache_mode
        self.has_seg = bool(has_seg)
        self.augment = self.is_train if augment is None else bool(augment)
        self.aug_mode = aug_mode
        self.modality_dropout_prob = float(modality_dropout_prob)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        if cache_mode not in ("mmap", "ram"):
            raise ValueError(f"cache_mode must be 'mmap' or 'ram', got '{cache_mode}'")
        if label_mode not in ("regions", "multiclass"):
            raise ValueError(f"label_mode must be 'regions' or 'multiclass', got '{label_mode}'")
        if aug_mode not in ("strong", "light", "none"):
            raise ValueError(f"aug_mode must be 'strong', 'light', or 'none', got '{aug_mode}'")

        self.samples: List[Dict[str, Any]] = []
        for cid in self.case_ids:
            img_path = self.data_dir / f"{cid}.npy"
            seg_path = self.data_dir / f"{cid}_seg.npy"
            pkl_path = self.data_dir / f"{cid}.pkl"
            if not img_path.is_file() or not pkl_path.is_file():
                raise FileNotFoundError(f"Missing .npy/.pkl for {cid} in {self.data_dir}")
            if self.has_seg and not seg_path.is_file():
                raise FileNotFoundError(f"Missing _seg.npy for {cid} in {self.data_dir}")
            self.samples.append(
                {
                    "case_id": cid,
                    "img_path": img_path,
                    "seg_path": seg_path if self.has_seg else None,
                    "pkl_path": pkl_path,
                }
            )

        self._ram_cache: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]] = {}
        if self.cache_mode == "ram":
            self._preload_into_ram(ram_preload_workers)

    def _load_meta(self, pkl_path: Path) -> Dict[str, Any]:
        with pkl_path.open("rb") as f:
            return pickle.load(f)

    def _preload_into_ram(self, num_workers: int) -> None:
        def _load_one(entry: Dict[str, Any]):
            img = np.load(entry["img_path"])
            seg = np.load(entry["seg_path"]) if entry["seg_path"] is not None else None
            meta = self._load_meta(entry["pkl_path"])
            return entry["case_id"], img, seg, meta

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
            for cid, img, seg, meta in pool.map(_load_one, self.samples):
                self._ram_cache[cid] = (img, seg, meta)

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_crop_start(
        self,
        spatial_shape: Tuple[int, int, int],
        meta: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Tuple[int, int, int]:
        """Choose 3D patch start coordinates with optional tumor-foreground oversampling."""
        px, py, pz = self.patch_size
        sx, sy, sz = spatial_shape
        max_starts = (max(0, sx - px), max(0, sy - py), max(0, sz - pz))

        class_locs: Dict[str, np.ndarray] = meta.get("class_locations", {})
        non_empty_keys = [
            k for k in ("label_1", "label_2", "label_3", "wt") if k in class_locs and len(class_locs[k]) > 0
        ]

        if non_empty_keys and rng.random() < self.fg_prob:
            chosen_key = non_empty_keys[int(rng.integers(0, len(non_empty_keys)))]
            coords = class_locs[chosen_key]
            center = coords[int(rng.integers(0, len(coords)))]
            # Add small jitter around the sampled tumor voxel so tumors are not always centered.
            jitter = rng.integers(-16, 17, size=3)
            starts = []
            for axis, (c, p, m) in enumerate(zip(center + jitter, self.patch_size, max_starts)):
                st = int(c - p // 2)
                starts.append(max(0, min(m, st)))
            return (starts[0], starts[1], starts[2])

        return tuple(int(rng.integers(0, m + 1)) if m > 0 else 0 for m in max_starts)  # type: ignore[return-value]

    @staticmethod
    def _build_affine_matrix_3d(
        rx: float, ry: float, rz: float, scale: float
    ) -> torch.Tensor:
        """Build 3x4 centered inverse affine matrix for `F.affine_grid`."""
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        mat = (rot_z @ rot_y @ rot_x) / float(scale)
        affine = np.zeros((3, 4), dtype=np.float32)
        affine[:, :3] = mat
        return torch.from_numpy(affine).unsqueeze(0)

    def _random_affine_3d(
        self,
        img: np.ndarray,
        seg: Optional[np.ndarray],
        rng: np.random.Generator,
        rotate_deg: float = 20.0,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        p_rot: float = 0.25,
        p_scale: float = 0.25,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply continuous 3D rotation and isotropic zoom via `grid_sample`."""
        do_rot = rng.random() < p_rot
        do_scale = rng.random() < p_scale
        if not do_rot and not do_scale:
            return img, seg

        max_rad = np.deg2rad(rotate_deg)
        rx, ry, rz = rng.uniform(-max_rad, max_rad, size=3) if do_rot else (0.0, 0.0, 0.0)
        scale = float(rng.uniform(*scale_range)) if do_scale else 1.0
        theta = self._build_affine_matrix_3d(float(rx), float(ry), float(rz), scale)

        img_t = torch.from_numpy(img).unsqueeze(0)
        grid = F.affine_grid(theta, size=img_t.shape, align_corners=False)
        img_out = F.grid_sample(
            img_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )[0].numpy()

        seg_out = seg
        if seg is not None:
            seg_t = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            seg_out = (
                F.grid_sample(
                    seg_t, grid, mode="nearest", padding_mode="zeros", align_corners=False
                )[0, 0]
                .numpy()
                .astype(np.uint8)
            )
        return img_out, seg_out

    @staticmethod
    def _gamma_foreground(
        vals: np.ndarray, gamma: float, invert: bool = False, retain_stats: bool = True
    ) -> np.ndarray:
        """Apply nnU-Net-style non-linear gamma augmentation on foreground voxels."""
        old_mean = float(vals.mean())
        old_std = float(vals.std())
        work = -vals if invert else vals
        vmin = float(work.min())
        vmax = float(work.max())
        span = max(vmax - vmin, 1e-8)
        normed = np.clip((work - vmin) / span, 0.0, 1.0)
        out = np.power(normed, gamma) * span + vmin
        if invert:
            out = -out
        if retain_stats:
            new_std = float(out.std())
            out = (out - float(out.mean())) / (new_std if new_std > 1e-8 else 1.0)
            out = out * (old_std if old_std > 1e-8 else 1.0) + old_mean
        return out.astype(np.float32, copy=False)

    def _augment_patch(
        self,
        img: np.ndarray,
        seg: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply 3D spatial & brain-masked intensity augmentations (`strong` or `light`)."""
        if self.aug_mode == "none":
            return img, seg

        # 1. Spatial transforms: continuous 3D rotate/zoom (strong) + flips + axial 90-deg rot
        if self.aug_mode == "strong":
            img, seg = self._random_affine_3d(img, seg, rng)

        for spatial_axis in (0, 1, 2):
            if rng.random() < 0.5:
                img = np.flip(img, axis=spatial_axis + 1)
                if seg is not None:
                    seg = np.flip(seg, axis=spatial_axis)

        if rng.random() < 0.5:
            k = int(rng.integers(1, 4))
            img = np.rot90(img, k=k, axes=(1, 2))
            if seg is not None:
                seg = np.rot90(seg, k=k, axes=(0, 1))

        img = np.ascontiguousarray(img, dtype=np.float32)
        if seg is not None:
            seg = np.ascontiguousarray(seg)

        # 2. Brain-foreground-only intensity augmentations (preserve background == 0.0)
        brain_mask = np.any(img != 0.0, axis=0)
        if np.any(brain_mask):
            n_ch = img.shape[0]
            if self.aug_mode == "strong":
                # Multiplicative brightness scaling (nnU-Net range: 0.75-1.25)
                if rng.random() < 0.2:
                    scale = rng.uniform(0.8, 1.2, size=(n_ch, 1)).astype(np.float32)
                    img[:, brain_mask] = img[:, brain_mask] * scale

                # Additive intensity shift
                if rng.random() < 0.2:
                    shift = rng.uniform(-0.1, 0.1, size=(n_ch, 1)).astype(np.float32)
                    img[:, brain_mask] = img[:, brain_mask] + shift

                # Contrast augmentation preserving per-channel range
                if rng.random() < 0.15:
                    for c in range(n_ch):
                        fg = img[c, brain_mask]
                        mn, mx, mu = float(fg.min()), float(fg.max()), float(fg.mean())
                        factor = float(rng.uniform(0.75, 1.25))
                        img[c, brain_mask] = np.clip((fg - mu) * factor + mu, mn, mx)

                # Non-linear Gamma transform (inverted + standard, retain_stats=True)
                if rng.random() < 0.1:
                    gamma = float(rng.uniform(0.7, 1.5))
                    for c in range(n_ch):
                        img[c, brain_mask] = self._gamma_foreground(
                            img[c, brain_mask], gamma=gamma, invert=True, retain_stats=True
                        )
                if rng.random() < 0.3:
                    gamma = float(rng.uniform(0.7, 1.5))
                    for c in range(n_ch):
                        img[c, brain_mask] = self._gamma_foreground(
                            img[c, brain_mask], gamma=gamma, invert=False, retain_stats=True
                        )

                # Gaussian blur (per-channel p=0.5 when triggered)
                if rng.random() < 0.2:
                    for c in range(n_ch):
                        if rng.random() < 0.5:
                            sigma = float(rng.uniform(0.5, 1.0))
                            img[c] = gaussian_filter(img[c], sigma=sigma)
                    img[:, ~brain_mask] = 0.0

                # Simulate low-resolution slices/scans
                if rng.random() < 0.15:
                    orig_spatial = img.shape[1:]
                    for c in range(n_ch):
                        if rng.random() < 0.5:
                            zoom = float(rng.uniform(0.5, 0.85))
                            down_shape = tuple(max(8, int(round(s * zoom))) for s in orig_spatial)
                            ch_t = torch.from_numpy(img[c]).unsqueeze(0).unsqueeze(0)
                            down = F.interpolate(ch_t, size=down_shape, mode="nearest")
                            up = F.interpolate(
                                down, size=orig_spatial, mode="trilinear", align_corners=False
                            )
                            img[c] = up[0, 0].numpy()
                    img[:, ~brain_mask] = 0.0
            else:
                # Light mode (MONAI / Swin-UNETR style)
                if rng.random() < 0.25:
                    scale = rng.uniform(0.9, 1.1, size=(n_ch, 1)).astype(np.float32)
                    img[:, brain_mask] = img[:, brain_mask] * scale
                if rng.random() < 0.25:
                    shift = rng.uniform(-0.1, 0.1, size=(n_ch, 1)).astype(np.float32)
                    img[:, brain_mask] = img[:, brain_mask] + shift

            # Gaussian noise on brain foreground
            if rng.random() < 0.2:
                sigma = float(rng.uniform(0.01, 0.08))
                n_fg = int(np.count_nonzero(brain_mask))
                noise = rng.normal(0.0, sigma, size=(n_ch, n_fg)).astype(np.float32)
                img[:, brain_mask] = img[:, brain_mask] + noise

        # 3. Optional modality dropout (drops 1 channel, keeps >= 3 active)
        if self.modality_dropout_prob > 0.0 and rng.random() < self.modality_dropout_prob:
            drop_ch = int(rng.integers(0, img.shape[0]))
            img[drop_ch] = 0.0

        return img, seg

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.samples[index]
        cid = entry["case_id"]

        if self.cache_mode == "ram":
            img_full, seg_full, meta = self._ram_cache[cid]
        else:
            img_full = np.load(entry["img_path"], mmap_mode="r")
            seg_full = np.load(entry["seg_path"], mmap_mode="r") if entry["seg_path"] else None
            meta = self._load_meta(entry["pkl_path"])

        if self.is_train:
            rng = self._rng
            spatial_shape = tuple(int(s) for s in img_full.shape[1:])
            x0, y0, z0 = self._choose_crop_start(spatial_shape, meta, rng)
            px, py, pz = self.patch_size
            x1 = min(spatial_shape[0], x0 + px)
            y1 = min(spatial_shape[1], y0 + py)
            z1 = min(spatial_shape[2], z0 + pz)

            img = np.array(img_full[:, x0:x1, y0:y1, z0:z1], dtype=np.float32, copy=True)
            seg = (
                np.array(seg_full[x0:x1, y0:y1, z0:z1], dtype=np.uint8, copy=True)
                if seg_full is not None
                else None
            )
            img, seg = pad_to_min_shape(img, seg, self.patch_size)
            if self.augment:
                img, seg = self._augment_patch(img, seg, rng)
        else:
            img = np.array(img_full, dtype=np.float32, copy=True)
            seg = np.array(seg_full, dtype=np.uint8, copy=True) if seg_full is not None else None

        sample: Dict[str, Any] = {
            "case_id": cid,
            "image": torch.from_numpy(img),
        }
        if seg is not None:
            sample["label"] = encode_seg_target(seg, self.label_mode)
            sample["seg_raw"] = torch.from_numpy(seg.astype(np.uint8))

        if not self.is_train:
            sample["meta"] = {
                "original_shape": meta["original_shape"],
                "cropped_shape": meta["cropped_shape"],
                "crop_bbox": meta["crop_bbox"],
                "spacing": meta["spacing"],
            }
        return sample


def _worker_init_fn(_worker_id: int) -> None:
    """Prevent OpenMP oversubscription and seed per-worker RNG deterministically."""
    torch.set_num_threads(1)
    base_seed = torch.initial_seed()
    worker_seed = base_seed % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        ds = getattr(worker_info.dataset, "dataset", worker_info.dataset)
        if hasattr(ds, "_rng"):
            ds._rng = np.random.default_rng(base_seed)


def build_dataloaders(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    split_json: Path | str = DEFAULT_SPLIT_JSON,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 2,
    num_workers: int = 4,
    cache_mode: str = "mmap",
    label_mode: str = "regions",
    aug_mode: str = "strong",
    fg_prob: float = 0.67,
    modality_dropout_prob: float = 0.0,
    fold: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
    include_test: bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create patient-disjoint train, val, and optional test DataLoaders."""
    data_dir = Path(data_dir)
    split_json = Path(split_json)

    train_ids = load_split_case_ids(split_json, "train", fold=fold)
    val_ids = load_split_case_ids(split_json, "val", fold=fold)

    with split_json.open("r", encoding="utf-8") as f:
        raw_split = json.load(f)
    has_test = "test" in raw_split
    test_ids = load_split_case_ids(split_json, "test", fold=fold) if has_test else []

    tr_pats = {parse_patient_id(c) for c in train_ids}
    va_pats = {parse_patient_id(c) for c in val_ids}
    te_pats = {parse_patient_id(c) for c in test_ids}
    assert len(tr_pats & va_pats) == 0, "Patient overlap between train and val!"
    assert len(tr_pats & te_pats) == 0, "Patient overlap between train and test!"
    assert len(va_pats & te_pats) == 0, "Patient overlap between val and test!"

    train_ds = BraTS23PreprocessedDataset(
        data_dir=data_dir,
        case_ids=train_ids,
        is_train=True,
        patch_size=patch_size,
        fg_prob=fg_prob,
        label_mode=label_mode,
        cache_mode=cache_mode,
        aug_mode=aug_mode,
        modality_dropout_prob=modality_dropout_prob,
        seed=seed,
    )
    val_ds = BraTS23PreprocessedDataset(
        data_dir=data_dir,
        case_ids=val_ids,
        is_train=False,
        label_mode=label_mode,
        cache_mode=cache_mode,
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _worker_init_fn if num_workers > 0 else None,
    }
    train_generator = torch.Generator()
    train_generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs)

    test_loader: Optional[DataLoader] = None
    if include_test:
        test_ds = BraTS23PreprocessedDataset(
            data_dir=data_dir,
            case_ids=test_ids if has_test else val_ids,
            is_train=False,
            label_mode=label_mode,
            cache_mode=cache_mode,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def run_sanity_check(args: argparse.Namespace) -> int:
    """Verify datasets, zero leakage, tensor shapes/values, and batch latency."""
    t0 = time.perf_counter()
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        split_json=args.split_json,
        patch_size=(args.patch_size, args.patch_size, args.patch_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_mode=args.cache_mode,
        label_mode=args.label_mode,
        aug_mode=args.aug_mode,
        fg_prob=args.fg_prob,
        pin_memory=torch.cuda.is_available(),
    )
    init_s = time.perf_counter() - t0
    print(
        f"[INIT] train={len(train_loader.dataset)} | val={len(val_loader.dataset)} | "
        f"test={len(test_loader.dataset)} | mode={args.cache_mode} | aug={args.aug_mode} | init={init_s:.2f}s"
    )

    # Measure a few training batches
    batch_times: List[float] = []
    fg_patches = 0
    total_patches = 0
    train_iter = iter(train_loader)
    for step in range(args.check_batches):
        tb = time.perf_counter()
        batch = next(train_iter)
        batch_times.append(time.perf_counter() - tb)

        img = batch["image"]
        lbl = batch["label"]
        seg_raw = batch["seg_raw"]
        assert torch.isfinite(img).all(), "Non-finite values in training image batch!"
        assert img.shape == (args.batch_size, 4, args.patch_size, args.patch_size, args.patch_size)
        if args.label_mode == "regions":
            assert lbl.shape == (args.batch_size, 3, args.patch_size, args.patch_size, args.patch_size)
        else:
            assert lbl.shape == (args.batch_size, args.patch_size, args.patch_size, args.patch_size)

        for b in range(args.batch_size):
            total_patches += 1
            if (seg_raw[b] > 0).any():
                fg_patches += 1

        if step == 0:
            nonzero = img[img != 0]
            print(
                f"[TRAIN BATCH 0] image={tuple(img.shape)} ({img.dtype}) | "
                f"label={tuple(lbl.shape)} ({lbl.dtype}) | "
                f"fg_mean={nonzero.mean().item():.3f}, fg_std={nonzero.std().item():.3f} | "
                f"raw_labels={torch.unique(seg_raw).tolist()}"
            )

    val_sample = next(iter(val_loader))
    test_sample = next(iter(test_loader))
    del train_iter
    print(
        f"[VAL SAMPLE]  id={val_sample['case_id'][0]} | image={tuple(val_sample['image'].shape)} | "
        f"label={tuple(val_sample['label'].shape)} | orig={val_sample['meta']['original_shape']}"
    )
    print(
        f"[TEST SAMPLE] id={test_sample['case_id'][0]} | image={tuple(test_sample['image'].shape)} | "
        f"label={tuple(test_sample['label'].shape)}"
    )
    steady_ms = (
        np.mean(batch_times[1:]) * 1000.0 if len(batch_times) > 1 else batch_times[0] * 1000.0
    )
    print(
        f"[SPEED] Warmup batch 0: {batch_times[0] * 1000:.1f} ms | "
        f"Steady-state avg ({args.batch_size} patches): {steady_ms:.1f} ms | "
        f"Patches containing tumor: {fg_patches}/{total_patches} ({fg_patches / total_patches * 100:.1f}%)"
    )
    print("[SANITY CHECK PASSED]")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BraTS 2023 preprocessed .npy Dataset & Sanity Check.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--cache-mode", choices=("mmap", "ram"), default="mmap")
    parser.add_argument("--label-mode", choices=("regions", "multiclass"), default="regions")
    parser.add_argument("--aug-mode", choices=("strong", "light", "none"), default="strong")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fg-prob", type=float, default=0.67)
    parser.add_argument("--check-batches", type=int, default=5)
    parser.add_argument("--sanity-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.sanity_check:
        raise SystemExit(run_sanity_check(cli_args))
    print("Import `build_dataloaders` or run with `--sanity-check` to verify.")
