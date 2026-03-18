from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import yaml
from numpy.lib import format as np_format


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_default_manifest(config: dict[str, Any], repo_root: Path) -> Path:
    manifest_rel = (
        config.get("paths", {})
        .get("manifests", {})
        .get("splits_single_secundo", "derivatives/preprocessing/splits_single_secundo.json")
    )
    return (repo_root / manifest_rel).resolve(strict=False)


def _read_npy_shape_from_npz(npz_path: Path, array_key: str) -> tuple[int, ...]:
    member_name = f"{array_key}.npy"
    with zipfile.ZipFile(npz_path, "r") as archive:
        try:
            with archive.open(member_name, "r") as handle:
                version = np_format.read_magic(handle)
                if version == (1, 0):
                    shape, _, _ = np_format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _, _ = np_format.read_array_header_2_0(handle)
                elif version == (3, 0):
                    shape, _, _ = np_format.read_array_header_3_0(handle)
                else:
                    raise ValueError(f"{npz_path}: unsupported NPY version {version} for key '{array_key}'")
        except KeyError as exc:
            raise KeyError(f"{npz_path} missing key '{array_key}'") from exc
    return tuple(int(dim) for dim in shape)


def _num_frames_from_shape(shape: tuple[int, ...], npz_path: Path, frames_key: str) -> int:
    if len(shape) not in (3, 4):
        raise ValueError(
            f"{npz_path}: '{frames_key}' must be [T,H,W] or [T,C,H,W], got shape {shape}"
        )
    return int(shape[0])


def _invalid_mask(
    archive: Any,
    t_len: int,
    npz_path: Path,
    labels_key: str | None,
    mask_key: str | None,
    exclude_label: int,
) -> np.ndarray:
    invalid = np.zeros(t_len, dtype=bool)
    if labels_key and labels_key in archive:
        labels = np.asarray(archive[labels_key]).squeeze()
        if labels.shape[0] != t_len:
            raise ValueError(f"{npz_path}: labels length {labels.shape[0]} != T {t_len}")
        invalid |= labels == exclude_label
    if mask_key and mask_key in archive:
        mask = np.asarray(archive[mask_key]).squeeze()
        if mask.shape[0] != t_len:
            raise ValueError(f"{npz_path}: mask length {mask.shape[0]} != T {t_len}")
        invalid |= mask.astype(bool)
    return invalid


def _count_valid_windows(
    t_len: int,
    *,
    window_size: int,
    pred_horizon: int,
    stride: int,
    invalid: np.ndarray | None = None,
) -> int:
    if t_len < window_size + pred_horizon:
        return 0

    if invalid is None:
        return int(((t_len - window_size - pred_horizon) // stride) + 1)

    n_windows = 0
    for end_ctx in range(window_size - 1, t_len - pred_horizon, stride):
        start_ctx = end_ctx - window_size + 1
        target_end = end_ctx + pred_horizon
        if start_ctx < 0 or target_end >= t_len:
            continue
        if invalid[start_ctx : target_end + 1].any():
            continue
        n_windows += 1
    return int(n_windows)


def load_manifest_cache(
    manifest_path: Path,
    *,
    frames_key: str,
    labels_key: str | None = None,
    mask_key: str | None = None,
    exclude_label: int = -1,
) -> dict[str, list[dict[str, Any]]]:
    manifest = _load_json(manifest_path)
    cache: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        entries: list[dict[str, Any]] = []
        for raw_path in manifest.get(split, []):
            npz_path = Path(raw_path).expanduser().resolve(strict=False)
            frame_shape = _read_npy_shape_from_npz(npz_path, frames_key)
            t_len = _num_frames_from_shape(frame_shape, npz_path=npz_path, frames_key=frames_key)
            invalid = None
            if labels_key or mask_key:
                with np.load(npz_path, allow_pickle=False) as archive:
                    invalid = _invalid_mask(
                        archive,
                        t_len=t_len,
                        npz_path=npz_path,
                        labels_key=labels_key,
                        mask_key=mask_key,
                        exclude_label=exclude_label,
                    )
            entries.append(
                {
                    "path": npz_path,
                    "frames": int(t_len),
                    "invalid": invalid,
                    "invalid_frames": int(invalid.sum()) if invalid is not None else 0,
                }
            )
        cache[split] = entries
    return cache


def count_windows_for_npz(
    npz_path: Path,
    *,
    window_size: int,
    pred_horizon: int,
    stride: int,
    frames_key: str,
    labels_key: str | None = None,
    mask_key: str | None = None,
    exclude_label: int = -1,
) -> dict[str, Any]:
    frame_shape = _read_npy_shape_from_npz(npz_path, frames_key)
    t_len = _num_frames_from_shape(frame_shape, npz_path=npz_path, frames_key=frames_key)
    invalid = None
    if labels_key or mask_key:
        with np.load(npz_path, allow_pickle=False) as archive:
            invalid = _invalid_mask(
                archive,
                t_len=t_len,
                npz_path=npz_path,
                labels_key=labels_key,
                mask_key=mask_key,
                exclude_label=exclude_label,
            )
    n_windows = _count_valid_windows(
        t_len,
        window_size=window_size,
        pred_horizon=pred_horizon,
        stride=stride,
        invalid=invalid,
    )

    return {
        "path": npz_path,
        "frames": int(t_len),
        "windows": int(n_windows),
        "invalid_frames": int(invalid.sum()),
    }


def summarize_manifest(
    manifest_path: Path,
    *,
    window_size: int,
    pred_horizon: int,
    stride: int,
    frames_key: str,
    labels_key: str | None = None,
    mask_key: str | None = None,
    exclude_label: int = -1,
) -> dict[str, list[dict[str, Any]]]:
    manifest = _load_json(manifest_path)
    summary: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        entries = []
        for raw_path in manifest.get(split, []):
            npz_path = Path(raw_path).expanduser().resolve(strict=False)
            entry = count_windows_for_npz(
                npz_path,
                window_size=window_size,
                pred_horizon=pred_horizon,
                stride=stride,
                frames_key=frames_key,
                labels_key=labels_key,
                mask_key=mask_key,
                exclude_label=exclude_label,
            )
            entries.append(entry)
        summary[split] = entries
    return summary


def print_summary(
    summary: dict[str, list[dict[str, Any]]],
    *,
    manifest_path: Path,
    window_size: int,
    pred_horizon: int,
    stride: int,
    frames_key: str,
    show_per_file: bool,
) -> None:
    print(f"Manifest: {manifest_path.as_posix()}")
    print(
        f"Parameters: window_size={window_size}, pred_horizon={pred_horizon}, "
        f"stride={stride}, frames_key='{frames_key}'"
    )
    print("")

    total_sessions = 0
    total_frames = 0
    total_windows = 0

    for split in ("train", "val", "test"):
        entries = summary.get(split, [])
        split_sessions = len(entries)
        split_frames = sum(item["frames"] for item in entries)
        split_windows = sum(item["windows"] for item in entries)
        total_sessions += split_sessions
        total_frames += split_frames
        total_windows += split_windows
        print(
            f"{split:>5}: sessions={split_sessions:>2} | "
            f"frames={split_frames:>6} | windows={split_windows:>6}"
        )
        if show_per_file:
            for item in entries:
                print(
                    f"      {item['path'].name}: "
                    f"frames={item['frames']}, windows={item['windows']}"
                )

    print("")
    print(
        f"Total: sessions={total_sessions} | frames={total_frames} | windows={total_windows}"
    )


def _resolve_stride(window_size: int, pred_horizon: int, stride_setting: int) -> int:
    if stride_setting == 1:
        return 1
    if stride_setting == 0:
        # Independent samples do not share any context or target frames.
        return window_size + pred_horizon
    raise ValueError("stride values must be 0 (independent) or 1 (advance by 1 frame)")


def build_combination_rows(
    manifest_cache: dict[str, list[dict[str, Any]]],
    *,
    window_lengths: list[int],
    horizon_lengths: list[int],
    strides: list[int],
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for window_size in window_lengths:
        for pred_horizon in horizon_lengths:
            for stride_setting in strides:
                effective_stride = _resolve_stride(window_size, pred_horizon, stride_setting)
                train_windows = sum(
                    _count_valid_windows(
                        item["frames"],
                        window_size=window_size,
                        pred_horizon=pred_horizon,
                        stride=effective_stride,
                        invalid=item["invalid"],
                    )
                    for item in manifest_cache.get("train", [])
                )
                val_windows = sum(
                    _count_valid_windows(
                        item["frames"],
                        window_size=window_size,
                        pred_horizon=pred_horizon,
                        stride=effective_stride,
                        invalid=item["invalid"],
                    )
                    for item in manifest_cache.get("val", [])
                )
                test_windows = sum(
                    _count_valid_windows(
                        item["frames"],
                        window_size=window_size,
                        pred_horizon=pred_horizon,
                        stride=effective_stride,
                        invalid=item["invalid"],
                    )
                    for item in manifest_cache.get("test", [])
                )
                rows.append(
                    {
                        "window_size": int(window_size),
                        "pred_horizon": int(pred_horizon),
                        "stride": int(stride_setting),
                        "train_windows": int(train_windows),
                        "val_windows": int(val_windows),
                        "test_windows": int(test_windows),
                        "total_windows": int(train_windows + val_windows + test_windows),
                    }
                )
    return rows


def print_combination_table(
    rows: list[dict[str, int]],
    *,
    manifest_path: Path,
    frames_key: str,
) -> None:
    print(f"Manifest: {manifest_path.as_posix()}")
    print("Stride convention: 0 = independent windows, 1 = move forward 1 frame at a time")
    print(f"Frames key: '{frames_key}'")
    print("")

    headers = [
        ("window", "window_size"),
        ("horizon", "pred_horizon"),
        ("stride", "stride"),
        ("train", "train_windows"),
        ("val", "val_windows"),
        ("test", "test_windows"),
        ("total", "total_windows"),
    ]
    widths: dict[str, int] = {}
    for label, key in headers:
        widths[key] = max(len(label), *(len(str(row[key])) for row in rows))

    header_line = " | ".join(label.rjust(widths[key]) for label, key in headers)
    separator = "-+-".join("-" * widths[key] for _, key in headers)
    print(header_line)
    print(separator)
    for row in rows:
        print(
            " | ".join(
                str(row[key]).rjust(widths[key])
                for _, key in headers
            )
        )


def save_rows_to_csv(rows: list[dict[str, int]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_size",
        "pred_horizon",
        "stride",
        "train_windows",
        "val_windows",
        "test_windows",
        "total_windows",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = _repo_root()
    config = _load_yaml(repo_root / "config" / "config.yml")
    default_csv_path = repo_root / "derivatives" / "modeling" / "window_counts.csv"
    parser = argparse.ArgumentParser(
        description="Count valid forecasting windows from a split manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_resolve_default_manifest(config, repo_root),
        help="Path to the split manifest JSON.",
    )
    parser.add_argument(
        "--window-lengths",
        nargs="+",
        type=int,
        default=[1, 5, 10, 15, 20, 40],
        help="Context window sizes to evaluate.",
    )
    parser.add_argument(
        "--horizon-lengths",
        nargs="+",
        type=int,
        default=[1, 5, 10, 15, 20, 40],
        help="Target horizon sizes to evaluate.",
    )
    parser.add_argument(
        "--strides",
        nargs="+",
        type=int,
        default=[0, 1],
        help="Stride modes to evaluate: 0 = independent windows, 1 = move forward 1 frame at a time.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv_path,
        help="Where to save the CSV summary table.",
    )
    parser.add_argument(
        "--frames-key",
        default=None,
        help="NPZ key containing frame data. Defaults to manifest['frames_key'] when available.",
    )
    parser.add_argument(
        "--labels-key",
        default=None,
        help="Optional NPZ key containing frame labels. Frames matching --exclude-label are skipped.",
    )
    parser.add_argument(
        "--mask-key",
        default=None,
        help="Optional NPZ key containing a boolean invalid-frame mask.",
    )
    parser.add_argument(
        "--exclude-label",
        type=int,
        default=-1,
        help="Label value treated as invalid when --labels-key is provided.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve(strict=False)
    manifest = _load_json(manifest_path)
    frames_key = args.frames_key or str(manifest.get("frames_key", "frames"))
    if any(v <= 0 for v in args.window_lengths):
        raise ValueError("window lengths must all be > 0")
    if any(v <= 0 for v in args.horizon_lengths):
        raise ValueError("horizon lengths must all be > 0")
    if any(v not in (0, 1) for v in args.strides):
        raise ValueError("strides must only contain 0 or 1")

    manifest_cache = load_manifest_cache(
        manifest_path,
        frames_key=frames_key,
        labels_key=args.labels_key,
        mask_key=args.mask_key,
        exclude_label=args.exclude_label,
    )
    rows = build_combination_rows(
        manifest_cache,
        window_lengths=args.window_lengths,
        horizon_lengths=args.horizon_lengths,
        strides=args.strides,
    )
    print_combination_table(
        rows,
        manifest_path=manifest_path,
        frames_key=frames_key,
    )
    csv_path = args.csv_path.expanduser().resolve(strict=False)
    save_rows_to_csv(rows, csv_path)
    print("")
    print(f"CSV saved to: {csv_path.as_posix()}")


if __name__ == "__main__":
    main()
