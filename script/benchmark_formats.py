#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark comparing I/O speed and storage size for 3D MRI volumes across formats:
  - npy  (.npy uncompressed + memory-mapped patch slicing)
  - npz  (.npz compressed archive, standard np.savez_compressed)
  - b2nd (.b2nd Blosc2 chunked N-dimensional array, nnU-Net v2 style)
  - h5   (.h5 HDF5 chunked raw and LZF-compressed)
  - zarr (.zarr Zarr chunked Blosc directory store)
  - pt   (.pt PyTorch tensor dict via torch.save/torch.load)
  - lmdb (LMDB single-file memory-mapped key-value store)

Usage:
  python benchmark_formats.py --n-cases 10 --n-repeat 3
  python benchmark_formats.py --formats npy npz b2nd h5 --keep-tmp
  python benchmark_formats.py --formats zarr pt lmdb --reuse-tmp
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_SOURCE_DIR = Path("/home/cuc.buithi/BRATS/data/fullres/train")
DEFAULT_WORK_DIR = Path("/home/cuc.buithi/BRATS/all/script/_bench_tmp")
DEFAULT_OUT_DIR = Path("/home/cuc.buithi/BRATS/all/script/output_brats23_analysis")

ALL_FORMATS = ("npy", "npz", "b2nd", "h5_raw", "h5_lzf", "zarr", "pt", "lmdb")


def optional_import(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


h5py = optional_import("h5py")
blosc2 = optional_import("blosc2")
zarr = optional_import("zarr")
torch = optional_import("torch")
lmdb = optional_import("lmdb")


def path_size_mb(path: Path) -> float:
    """Compute total disk size (MB) of a file or directory (e.g., .zarr store)."""
    if path.is_file():
        return path.stat().st_size / (1024.0 * 1024.0)
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024.0 * 1024.0)
    return 0.0


def find_source_cases(source_dir: Path, n_cases: int) -> List[Tuple[str, Path, Path]]:
    """Find `n_cases` preprocessed BraTS cases (`<id>.npy` + `<id>_seg.npy`)."""
    cases: List[Tuple[str, Path, Path]] = []
    for npy_path in sorted(source_dir.glob("*.npy")):
        if npy_path.name.endswith("_seg.npy"):
            continue
        seg_path = npy_path.with_name(npy_path.stem + "_seg.npy")
        if seg_path.is_file():
            cases.append((npy_path.stem, npy_path, seg_path))
        if len(cases) >= n_cases:
            break
    return cases


def compute_patch_slices(
    shape: Tuple[int, ...], patch_size: int, rng: random.Random
) -> Tuple[slice, slice, slice]:
    """Sample a 3D spatial crop `(sx, sy, sz)` safely within `(D, H, W)`."""
    _, d, h, w = shape
    pd, ph, pw = min(patch_size, d), min(patch_size, h), min(patch_size, w)
    z0 = rng.randint(0, max(0, d - pd))
    y0 = rng.randint(0, max(0, h - ph))
    x0 = rng.randint(0, max(0, w - pw))
    return slice(z0, z0 + pd), slice(y0, y0 + ph), slice(x0, x0 + pw)


# ---------------------------------------------------------------------------
# Write / Read implementations for each format
# ---------------------------------------------------------------------------

def write_npy(fmt_dir: Path, case_id: str, img: np.ndarray, seg: np.ndarray) -> List[Path]:
    p_img = fmt_dir / f"{case_id}.npy"
    p_seg = fmt_dir / f"{case_id}_seg.npy"
    np.save(p_img, img)
    np.save(p_seg, seg)
    return [p_img, p_seg]


def read_full_npy(fmt_dir: Path, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    img = np.load(fmt_dir / f"{case_id}.npy")
    seg = np.load(fmt_dir / f"{case_id}_seg.npy")
    return img, seg


def read_patch_npy(
    fmt_dir: Path, case_id: str, sl: Tuple[slice, slice, slice]
) -> Tuple[np.ndarray, np.ndarray]:
    img_mm = np.load(fmt_dir / f"{case_id}.npy", mmap_mode="r")
    seg_mm = np.load(fmt_dir / f"{case_id}_seg.npy", mmap_mode="r")
    img = np.array(img_mm[:, sl[0], sl[1], sl[2]])
    seg = np.array(seg_mm[:, sl[0], sl[1], sl[2]])
    return img, seg


def write_npz(fmt_dir: Path, case_id: str, img: np.ndarray, seg: np.ndarray) -> List[Path]:
    p = fmt_dir / f"{case_id}.npz"
    np.savez_compressed(p, data=img, seg=seg)
    return [p]


def read_full_npz(fmt_dir: Path, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(fmt_dir / f"{case_id}.npz") as d:
        return d["data"], d["seg"]


def read_patch_npz(
    fmt_dir: Path, case_id: str, sl: Tuple[slice, slice, slice]
) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(fmt_dir / f"{case_id}.npz") as d:
        img = np.array(d["data"][:, sl[0], sl[1], sl[2]])
        seg = np.array(d["seg"][:, sl[0], sl[1], sl[2]])
    return img, seg


def _chunk_shape(shape: Tuple[int, ...], base: int = 64) -> Tuple[int, ...]:
    c, d, h, w = shape
    return (1, min(base, d), min(base, h), min(base, w))


def _block_shape(chunk: Tuple[int, ...], base: int = 32) -> Tuple[int, ...]:
    return (1, min(base, chunk[1]), min(base, chunk[2]), min(base, chunk[3]))


def write_b2nd(fmt_dir: Path, case_id: str, img: np.ndarray, seg: np.ndarray) -> List[Path]:
    assert blosc2 is not None
    p_img = fmt_dir / f"{case_id}.b2nd"
    p_seg = fmt_dir / f"{case_id}_seg.b2nd"
    c_img = _chunk_shape(img.shape)
    c_seg = _chunk_shape(seg.shape)
    cparams = blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=3)
    blosc2.asarray(
        img,
        urlpath=str(p_img),
        chunks=c_img,
        blocks=_block_shape(c_img),
        cparams=cparams,
        mode="w",
    )
    blosc2.asarray(
        seg,
        urlpath=str(p_seg),
        chunks=c_seg,
        blocks=_block_shape(c_seg),
        cparams=cparams,
        mode="w",
    )
    return [p_img, p_seg]


def read_full_b2nd(fmt_dir: Path, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    assert blosc2 is not None
    img = blosc2.open(str(fmt_dir / f"{case_id}.b2nd"), mode="r")[:]
    seg = blosc2.open(str(fmt_dir / f"{case_id}_seg.b2nd"), mode="r")[:]
    return img, seg


def read_patch_b2nd(
    fmt_dir: Path, case_id: str, sl: Tuple[slice, slice, slice]
) -> Tuple[np.ndarray, np.ndarray]:
    assert blosc2 is not None
    img_b = blosc2.open(str(fmt_dir / f"{case_id}.b2nd"), mode="r")
    seg_b = blosc2.open(str(fmt_dir / f"{case_id}_seg.b2nd"), mode="r")
    return img_b[:, sl[0], sl[1], sl[2]], seg_b[:, sl[0], sl[1], sl[2]]


def _write_h5(
    fmt_dir: Path, case_id: str, img: np.ndarray, seg: np.ndarray, compression: Optional[str]
) -> List[Path]:
    assert h5py is not None
    p = fmt_dir / f"{case_id}.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset(
            "image", data=img, chunks=_chunk_shape(img.shape), compression=compression
        )
        f.create_dataset(
            "seg", data=seg, chunks=_chunk_shape(seg.shape), compression=compression
        )
    return [p]


def write_h5_raw(fmt_dir: Path, case_id: str, img: np.ndarray, seg: np.ndarray) -> List[Path]:
    return _write_h5(fmt_dir, case_id, img, seg, compression=None)


def write_h5_lzf(fmt_dir: Path, case_id: str, img: np.ndarray, seg: np.ndarray) -> List[Path]:
    return _write_h5(fmt_dir, case_id, img, seg, compression="lzf")


def read_full_h5(fmt_dir: Path, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    assert h5py is not None
    with h5py.File(fmt_dir / f"{case_id}.h5", "r") as f:
        return f["image"][:], f["seg"][:]


def read_patch_h5(
    fmt_dir: Path, case_id: str, sl: Tuple[slice, slice, slice]
) -> Tuple[np.ndarray, np.ndarray]:
    assert h5py is not None
    with h5py.File(fmt_dir / f"{case_id}.h5", "r") as f:
        img = f["image"][:, sl[0], sl[1], sl[2]]
        seg = f["seg"][:, sl[0], sl[1], sl[2]]
    return img, seg


def write_zarr(fmt_dir: Path, case_id: str, img: np.ndarray, seg: np.ndarray) -> List[Path]:
    assert zarr is not None
    p = fmt_dir / f"{case_id}.zarr"
    if p.exists():
        shutil.rmtree(p)
    grp = zarr.open_group(str(p), mode="w")
    grp.create_dataset("image", data=img, chunks=_chunk_shape(img.shape))
    grp.create_dataset("seg", data=seg, chunks=_chunk_shape(seg.shape))
    return [p]


def read_full_zarr(fmt_dir: Path, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    assert zarr is not None
    grp = zarr.open_group(str(fmt_dir / f"{case_id}.zarr"), mode="r")
    return grp["image"][:], grp["seg"][:]


def read_patch_zarr(
    fmt_dir: Path, case_id: str, sl: Tuple[slice, slice, slice]
) -> Tuple[np.ndarray, np.ndarray]:
    assert zarr is not None
    grp = zarr.open_group(str(fmt_dir / f"{case_id}.zarr"), mode="r")
    img = grp["image"][:, sl[0], sl[1], sl[2]]
    seg = grp["seg"][:, sl[0], sl[1], sl[2]]
    return img, seg


def write_pt(fmt_dir: Path, case_id: str, img: np.ndarray, seg: np.ndarray) -> List[Path]:
    assert torch is not None
    p = fmt_dir / f"{case_id}.pt"
    torch.save({"image": torch.from_numpy(img), "seg": torch.from_numpy(seg)}, p)
    return [p]


def read_full_pt(fmt_dir: Path, case_id: str) -> Tuple[Any, Any]:
    assert torch is not None
    d = torch.load(fmt_dir / f"{case_id}.pt", map_location="cpu", weights_only=True)
    return d["image"], d["seg"]


def read_patch_pt(
    fmt_dir: Path, case_id: str, sl: Tuple[slice, slice, slice]
) -> Tuple[Any, Any]:
    assert torch is not None
    d = torch.load(fmt_dir / f"{case_id}.pt", map_location="cpu", weights_only=True)
    img = d["image"][:, sl[0], sl[1], sl[2]].clone()
    seg = d["seg"][:, sl[0], sl[1], sl[2]].clone()
    return img, seg


def benchmark_lmdb(
    cases: List[Tuple[str, np.ndarray, np.ndarray]],
    case_patches: Dict[str, Tuple[slice, slice, slice]],
    work_dir: Path,
    n_repeat: int,
    reuse_tmp: bool,
) -> Dict[str, Any]:
    """Benchmark LMDB single-store key-value mmap I/O (zero-copy buffer slicing)."""
    assert lmdb is not None
    fmt_dir = work_dir / "lmdb"
    db_path = fmt_dir / "dataset.lmdb"
    fmt_dir.mkdir(parents=True, exist_ok=True)

    write_times_ms: List[float] = []
    total_raw_bytes = sum(img.nbytes + seg.nbytes for _, img, seg in cases)
    if not reuse_tmp or not db_path.exists():
        if db_path.exists():
            shutil.rmtree(db_path)
        map_size = max(2 * 1024**3, total_raw_bytes * 3)
        env = lmdb.open(str(db_path), map_size=map_size)
        for cid, img, seg in cases:
            meta = json.dumps(
                {
                    "img_shape": list(img.shape),
                    "img_dtype": str(img.dtype),
                    "seg_shape": list(seg.shape),
                    "seg_dtype": str(seg.dtype),
                }
            ).encode("utf-8")
            t0 = time.perf_counter()
            with env.begin(write=True) as txn:
                txn.put(f"{cid}:meta".encode("utf-8"), meta)
                txn.put(f"{cid}:img".encode("utf-8"), img.tobytes())
                txn.put(f"{cid}:seg".encode("utf-8"), seg.tobytes())
            write_times_ms.append((time.perf_counter() - t0) * 1000.0)
        env.sync()
        env.close()

    disk_mb_total = path_size_mb(db_path)
    disk_mb_mean = disk_mb_total / len(cases)
    raw_mb_mean = float(
        statistics.mean((img.nbytes + seg.nbytes) / (1024.0 * 1024.0) for _, img, seg in cases)
    )

    env_r = lmdb.open(str(db_path), readonly=True, lock=False, readahead=False)
    try:
        full_times_ms: List[float] = []
        for _ in range(n_repeat):
            for cid, _, _ in cases:
                t0 = time.perf_counter()
                with env_r.begin(buffers=True) as txn:
                    meta = json.loads(bytes(txn.get(f"{cid}:meta".encode("utf-8"))).decode("utf-8"))
                    img_r = np.frombuffer(
                        txn.get(f"{cid}:img".encode("utf-8")), dtype=meta["img_dtype"]
                    ).reshape(meta["img_shape"]).copy()
                    seg_r = np.frombuffer(
                        txn.get(f"{cid}:seg".encode("utf-8")), dtype=meta["seg_dtype"]
                    ).reshape(meta["seg_shape"]).copy()
                _ = float(img_r[0, 0, 0, 0]) + float(seg_r[0, 0, 0, 0])
                full_times_ms.append((time.perf_counter() - t0) * 1000.0)

        patch_times_ms: List[float] = []
        for _ in range(n_repeat):
            for cid, _, _ in cases:
                sl = case_patches[cid]
                t0 = time.perf_counter()
                with env_r.begin(buffers=True) as txn:
                    meta = json.loads(bytes(txn.get(f"{cid}:meta".encode("utf-8"))).decode("utf-8"))
                    img_v = np.frombuffer(
                        txn.get(f"{cid}:img".encode("utf-8")), dtype=meta["img_dtype"]
                    ).reshape(meta["img_shape"])
                    seg_v = np.frombuffer(
                        txn.get(f"{cid}:seg".encode("utf-8")), dtype=meta["seg_dtype"]
                    ).reshape(meta["seg_shape"])
                    img_p = np.array(img_v[:, sl[0], sl[1], sl[2]])
                    seg_p = np.array(seg_v[:, sl[0], sl[1], sl[2]])
                _ = float(img_p[0, 0, 0, 0]) + float(seg_p[0, 0, 0, 0])
                patch_times_ms.append((time.perf_counter() - t0) * 1000.0)
    finally:
        env_r.close()

    return {
        "format": "lmdb",
        "description": "LMDB single-store mmap (zero-copy patch slice)",
        "n_cases": len(cases),
        "n_repeat": n_repeat,
        "raw_ram_mb_mean": raw_mb_mean,
        "disk_mb_mean": disk_mb_mean,
        "compression_ratio": raw_mb_mean / disk_mb_mean if disk_mb_mean > 0 else 0.0,
        "write_ms_mean": float(statistics.mean(write_times_ms)) if write_times_ms else None,
        "full_read_ms_mean": float(statistics.mean(full_times_ms)),
        "full_read_ms_median": float(statistics.median(full_times_ms)),
        "patch_read_ms_mean": float(statistics.mean(patch_times_ms)),
        "patch_read_ms_median": float(statistics.median(patch_times_ms)),
    }


FORMAT_SPECS: Dict[str, Dict[str, Any]] = {
    "npy": {
        "available": lambda: True,
        "write": write_npy,
        "read_full": read_full_npy,
        "read_patch": read_patch_npy,
        "paths": lambda d, cid: [d / f"{cid}.npy", d / f"{cid}_seg.npy"],
        "desc": "Uncompressed .npy (+ mmap_mode='r' patch)",
    },
    "npz": {
        "available": lambda: True,
        "write": write_npz,
        "read_full": read_full_npz,
        "read_patch": read_patch_npz,
        "paths": lambda d, cid: [d / f"{cid}.npz"],
        "desc": "Compressed .npz (np.savez_compressed)",
    },
    "b2nd": {
        "available": lambda: blosc2 is not None,
        "write": write_b2nd,
        "read_full": read_full_b2nd,
        "read_patch": read_patch_b2nd,
        "paths": lambda d, cid: [d / f"{cid}.b2nd", d / f"{cid}_seg.b2nd"],
        "desc": "Blosc2 .b2nd (ZSTD level 3, chunk 64³)",
    },
    "h5_raw": {
        "available": lambda: h5py is not None,
        "write": write_h5_raw,
        "read_full": read_full_h5,
        "read_patch": read_patch_h5,
        "paths": lambda d, cid: [d / f"{cid}.h5"],
        "desc": "HDF5 .h5 (uncompressed, chunk 64³)",
    },
    "h5_lzf": {
        "available": lambda: h5py is not None,
        "write": write_h5_lzf,
        "read_full": read_full_h5,
        "read_patch": read_patch_h5,
        "paths": lambda d, cid: [d / f"{cid}.h5"],
        "desc": "HDF5 .h5 (LZF compressed, chunk 64³)",
    },
    "zarr": {
        "available": lambda: zarr is not None,
        "write": write_zarr,
        "read_full": read_full_zarr,
        "read_patch": read_patch_zarr,
        "paths": lambda d, cid: [d / f"{cid}.zarr"],
        "desc": "Zarr .zarr (Blosc default, chunk 64³)",
    },
    "pt": {
        "available": lambda: torch is not None,
        "write": write_pt,
        "read_full": read_full_pt,
        "read_patch": read_patch_pt,
        "paths": lambda d, cid: [d / f"{cid}.pt"],
        "desc": "PyTorch .pt tensor dict (torch.save/load)",
    },
    "lmdb": {
        "available": lambda: lmdb is not None,
        "desc": "LMDB single-store mmap (zero-copy patch slice)",
    },
}


def benchmark_format(
    fmt: str,
    cases: List[Tuple[str, np.ndarray, np.ndarray]],
    case_patches: Dict[str, Tuple[slice, slice, slice]],
    work_dir: Path,
    n_repeat: int,
    reuse_tmp: bool,
) -> Dict[str, Any]:
    """Run write, disk-size, full-volume read, and 3D patch read benchmarks."""
    if fmt == "lmdb":
        return benchmark_lmdb(cases, case_patches, work_dir, n_repeat, reuse_tmp)
    spec = FORMAT_SPECS[fmt]
    fmt_dir = work_dir / fmt
    fmt_dir.mkdir(parents=True, exist_ok=True)

    write_times_ms: List[float] = []
    sizes_mb: List[float] = []

    for cid, img, seg in cases:
        if not reuse_tmp or not all(p.exists() for p in spec["paths"](fmt_dir, cid)):
            t0 = time.perf_counter()
            paths = spec["write"](fmt_dir, cid, img, seg)
            write_times_ms.append((time.perf_counter() - t0) * 1000.0)
        else:
            paths = spec["paths"](fmt_dir, cid)
        sizes_mb.append(sum(path_size_mb(p) for p in paths))

    # Full-volume read benchmark
    full_times_ms: List[float] = []
    for rep in range(n_repeat):
        for cid, _, _ in cases:
            t0 = time.perf_counter()
            img_r, seg_r = spec["read_full"](fmt_dir, cid)
            _ = float(img_r[0, 0, 0, 0]) + float(seg_r[0, 0, 0, 0])
            full_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # 3D patch read benchmark
    patch_times_ms: List[float] = []
    for rep in range(n_repeat):
        for cid, _, _ in cases:
            sl = case_patches[cid]
            t0 = time.perf_counter()
            img_p, seg_p = spec["read_patch"](fmt_dir, cid, sl)
            _ = float(img_p[0, 0, 0, 0]) + float(seg_p[0, 0, 0, 0])
            patch_times_ms.append((time.perf_counter() - t0) * 1000.0)

    raw_mb_mean = float(
        statistics.mean((img.nbytes + seg.nbytes) / (1024.0 * 1024.0) for _, img, seg in cases)
    )
    disk_mb_mean = float(statistics.mean(sizes_mb))

    return {
        "format": fmt,
        "description": spec["desc"],
        "n_cases": len(cases),
        "n_repeat": n_repeat,
        "raw_ram_mb_mean": raw_mb_mean,
        "disk_mb_mean": disk_mb_mean,
        "compression_ratio": raw_mb_mean / disk_mb_mean if disk_mb_mean > 0 else 0.0,
        "write_ms_mean": float(statistics.mean(write_times_ms)) if write_times_ms else None,
        "full_read_ms_mean": float(statistics.mean(full_times_ms)),
        "full_read_ms_median": float(statistics.median(full_times_ms)),
        "patch_read_ms_mean": float(statistics.mean(patch_times_ms)),
        "patch_read_ms_median": float(statistics.median(patch_times_ms)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark npy, npz, b2nd, h5, zarr on BraTS 2023 3D volumes."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-cases", type=int, default=8)
    parser.add_argument("--n-repeat", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(ALL_FORMATS),
        choices=list(ALL_FORMATS),
        help="Formats to benchmark.",
    )
    parser.add_argument(
        "--reuse-tmp",
        action="store_true",
        help="Reuse existing temp files in --work-dir if already written.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary benchmark files after finishing.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="benchmark",
        help="Output filename suffix tag (e.g., brats_env or zarr_env).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    src_cases = find_source_cases(args.source_dir, args.n_cases)
    if not src_cases:
        raise RuntimeError(f"No .npy + _seg.npy pairs found in: {args.source_dir}")

    print(f"[INFO] Loading {len(src_cases)} real BraTS23 cases into memory for fair benchmark...")
    loaded_cases: List[Tuple[str, np.ndarray, np.ndarray]] = []
    rng = random.Random(args.seed)
    case_patches: Dict[str, Tuple[slice, slice, slice]] = {}
    for cid, img_p, seg_p in src_cases:
        img = np.load(img_p)
        seg = np.load(seg_p)
        loaded_cases.append((cid, img, seg))
        case_patches[cid] = compute_patch_slices(img.shape, args.patch_size, rng)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for fmt in args.formats:
        spec = FORMAT_SPECS[fmt]
        if not spec["available"]():
            print(f"[SKIP] {fmt}: required library not installed in this environment.")
            continue
        print(f"[RUN] Benchmarking {fmt} ({spec['desc']})...")
        res = benchmark_format(
            fmt=fmt,
            cases=loaded_cases,
            case_patches=case_patches,
            work_dir=args.work_dir,
            n_repeat=args.n_repeat,
            reuse_tmp=args.reuse_tmp,
        )
        results.append(res)

    if not args.keep_tmp and args.work_dir.exists():
        shutil.rmtree(args.work_dir)

    print("\n" + "=" * 96)
    print(
        f"{'Format':<8} | {'Disk(MB)':>8} | {'Ratio':>6} | {'Write(ms)':>10} | "
        f"{'Full Med(ms)':>12} | {'Patch Med(ms)':>13} | {'Patch Mean(ms)':>14}"
    )
    print("-" * 96)
    for r in sorted(results, key=lambda x: x["patch_read_ms_median"]):
        w_str = f"{r['write_ms_mean']:.1f}" if r["write_ms_mean"] is not None else "reused"
        print(
            f"{r['format']:<8} | {r['disk_mb_mean']:>8.2f} | {r['compression_ratio']:>5.2f}x | "
            f"{w_str:>10} | {r['full_read_ms_median']:>12.2f} | "
            f"{r['patch_read_ms_median']:>13.2f} | {r['patch_read_ms_mean']:>14.2f}"
        )
    print("=" * 96)

    out_json = args.out_dir / f"format_benchmark_{args.tag}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] Saved benchmark JSON to: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
