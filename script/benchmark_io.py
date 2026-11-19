#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-contained I/O speed benchmark for preprocessed data inside `/home/cuc.buithi/BRATS/all/data`.

Compares reading modes on the same BraTS 2023 GLI cases:
1. RAM cache slice (128x128x128 patch from in-memory array)
2. /dev/shm .npy mmap_mode='r' (128x128x128 patch crop)
3. NFS .npy mmap_mode='r' (128x128x128 patch crop)
4. /dev/shm .npy full volume read
5. NFS .npy full volume read
6. Compressed .npz full volume read (temporary compressed copies, cleaned up after test)

Saves structured results to `output_brats23_analysis/io_benchmark.json` and `io_benchmark.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


DEFAULT_DATA_DIR = Path("/home/cuc.buithi/BRATS/all/data")
DEFAULT_OUT_DIR = Path("/home/cuc.buithi/BRATS/all/script/output_brats23_analysis")


def collect_cases(data_dir: Path, n_cases: int) -> List[Tuple[Path, Path, Path]]:
    """Collect up to `n_cases` preprocessed (image.npy, seg.npy, meta.pkl) tuples."""
    img_files = sorted(
        p for p in data_dir.glob("*.npy") if not p.name.endswith("_seg.npy")
    )
    cases: List[Tuple[Path, Path, Path]] = []
    for img_path in img_files:
        stem = img_path.stem
        seg_path = data_dir / f"{stem}_seg.npy"
        pkl_path = data_dir / f"{stem}.pkl"
        if seg_path.is_file() and pkl_path.is_file():
            cases.append((img_path, seg_path, pkl_path))
        if len(cases) >= n_cases:
            break
    return cases


def patch_slices(shape: Tuple[int, int, int], patch_size: int) -> Tuple[slice, slice, slice]:
    """Compute centered crop slices up to `patch_size` along each spatial axis."""
    slices = []
    for dim in shape:
        size = min(dim, patch_size)
        start = max(0, (dim - size) // 2)
        slices.append(slice(start, start + size))
    return tuple(slices)  # type: ignore[return-value]


def time_loader(
    fn: Callable[[Any], Any], items: List[Any], n_repeat: int
) -> Dict[str, float]:
    """Benchmark `fn` across items taking best-of-`n_repeat` per item."""
    best_times: List[float] = []
    for item in items:
        runs: List[float] = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn(item)
            runs.append(time.perf_counter() - t0)
        best_times.append(min(runs))

    mean_s = statistics.mean(best_times)
    median_s = statistics.median(best_times)
    std_s = statistics.stdev(best_times) if len(best_times) > 1 else 0.0
    return {
        "n_cases": len(best_times),
        "mean_ms": round(mean_s * 1000.0, 3),
        "median_ms": round(median_s * 1000.0, 3),
        "std_ms": round(std_s * 1000.0, 3),
        "total_s": round(sum(best_times), 4),
    }


def load_pkl(p: Path) -> Any:
    """Load pickle metadata with explicit file close."""
    with p.open("rb") as f:
        return pickle.load(f)


def write_markdown(summary: Dict[str, Any], out_path: Path) -> None:
    """Save benchmark comparison table as Markdown."""
    rows = sorted(summary["results"].items(), key=lambda kv: kv[1]["mean_ms"])
    fastest = rows[0][1]["mean_ms"]
    slowest = rows[-1][1]["mean_ms"]

    lines = [
        "# Benchmark Tốc Độ Đọc Dữ Liệu (`all/data`)",
        "",
        f"- **Thư mục dữ liệu:** `{summary['data_dir']}`",
        f"- **Số ca đo:** `{summary['n_cases']}` ca | **Lặp lại:** `{summary['n_repeat']}` lần/ca (lấy best-of-n) | **Patch size:** `{summary['patch_size']}^3`",
        "",
        "| Hạng | Phương thức đọc | Mean (ms/ca) | Median (ms/ca) | Std (ms) | Nhanh hơn `.npz` |",
        "| :---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for idx, (name, stat) in enumerate(rows, 1):
        speedup = slowest / stat["mean_ms"]
        rel = stat["mean_ms"] / fastest
        lines.append(
            f"| #{idx} | {name} | **{stat['mean_ms']:.2f}** | {stat['median_ms']:.2f} | "
            f"{stat['std_ms']:.2f} | {speedup:.2f}x (x{rel:.2f} vs #1) |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark I/O read speeds on preprocessed .npy files in all/data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--n-repeat", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(data_dir, args.n_cases)
    if not cases:
        raise RuntimeError(f"No preprocessed .npy/.pkl cases found in {data_dir}")

    print(f"[INFO] Benchmarking {len(cases)} cases from {data_dir} (patch={args.patch_size}^3, repeat={args.n_repeat})")

    # Pre-load RAM cache items
    ram_items: List[Tuple[np.ndarray, np.ndarray, Tuple[slice, slice, slice]]] = []
    nfs_patch_items: List[Tuple[Path, Path, Path, Tuple[slice, slice, slice]]] = []
    for img_p, seg_p, pkl_p in cases:
        img = np.load(img_p)
        seg = np.load(seg_p)
        sl = patch_slices(seg.shape, args.patch_size)
        ram_items.append((img, seg, sl))
        nfs_patch_items.append((img_p, seg_p, pkl_p, sl))

    results: Dict[str, Dict[str, float]] = {}

    results["RAM cache slice (patch)"] = time_loader(
        lambda it: (np.array(it[0][:, it[2][0], it[2][1], it[2][2]]), np.array(it[1][it[2]])),
        ram_items,
        args.n_repeat,
    )

    results["NFS .npy mmap (patch)"] = time_loader(
        lambda it: (
            np.array(np.load(it[0], mmap_mode="r")[:, it[3][0], it[3][1], it[3][2]]),
            np.array(np.load(it[1], mmap_mode="r")[it[3]]),
            load_pkl(it[2]),
        ),
        nfs_patch_items,
        args.n_repeat,
    )

    results["NFS .npy full read"] = time_loader(
        lambda it: (np.load(it[0]), np.load(it[1]), load_pkl(it[2])),
        cases,
        args.n_repeat,
    )

    # Benchmark /dev/shm (.npy) and temporary .npz compressed on NFS (cleaned up immediately)
    shm_dir = Path(tempfile.mkdtemp(prefix="bench_brats_shm_", dir="/dev/shm"))
    npz_dir = Path(tempfile.mkdtemp(prefix="bench_brats_npz_", dir=str(out_dir)))
    try:
        shm_items = []
        npz_items = []
        for (img_p, seg_p, pkl_p), (img, seg, sl) in zip(cases, ram_items):
            sh_img = shm_dir / img_p.name
            sh_seg = shm_dir / seg_p.name
            sh_pkl = shm_dir / pkl_p.name
            shutil.copy2(img_p, sh_img)
            shutil.copy2(seg_p, sh_seg)
            shutil.copy2(pkl_p, sh_pkl)
            shm_items.append((sh_img, sh_seg, sh_pkl, sl))

            npz_p = npz_dir / f"{img_p.stem}.npz"
            np.savez_compressed(npz_p, data=img, seg=seg)
            npz_items.append(npz_p)

        results["/dev/shm .npy mmap (patch)"] = time_loader(
            lambda it: (
                np.array(np.load(it[0], mmap_mode="r")[:, it[3][0], it[3][1], it[3][2]]),
                np.array(np.load(it[1], mmap_mode="r")[it[3]]),
                load_pkl(it[2]),
            ),
            shm_items,
            args.n_repeat,
        )

        results["/dev/shm .npy full read"] = time_loader(
            lambda it: (np.load(it[0]), np.load(it[1]), load_pkl(it[2])),
            shm_items,
            args.n_repeat,
        )

        def read_npz(p: Path):
            with np.load(p) as d:
                return np.array(d["data"]), np.array(d["seg"])

        results["NFS .npz compressed (full)"] = time_loader(read_npz, npz_items, args.n_repeat)
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
        shutil.rmtree(npz_dir, ignore_errors=True)

    summary = {
        "data_dir": str(data_dir),
        "n_cases": len(cases),
        "n_repeat": args.n_repeat,
        "patch_size": args.patch_size,
        "results": results,
    }

    json_path = out_dir / "io_benchmark.json"
    md_path = out_dir / "io_benchmark.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_markdown(summary, md_path)

    print("\n[RESULTS] Sorted fastest -> slowest:")
    for name, stat in sorted(results.items(), key=lambda kv: kv[1]["mean_ms"]):
        print(f"  - {name:<28s}: {stat['mean_ms']:7.2f} ms/case (median={stat['median_ms']:7.2f} ms)")
    print(f"\n[DONE] Saved to {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
