from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .io import (
    STAGE_REORIENTED_RESIZED,
    derive_session_id_from_path,
    load_stage_npz,
    save_stage_npz,
)


def _validate_frames_thw(frames: np.ndarray, context: str = "frames") -> np.ndarray:
    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError(f"{context}: expected shape (T,H,W), got {arr.shape}")
    return arr


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
        return default
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return default


def _select_preview_indices(T: int, n_frames: int) -> np.ndarray:
    if T <= 0:
        return np.array([], dtype=np.int64)
    n = max(1, min(int(n_frames), int(T)))
    if n == 1:
        return np.array([0], dtype=np.int64)
    idxs = np.linspace(0, T - 1, num=n, dtype=np.int64)
    # Ensure strictly increasing unique indices while preserving order.
    return np.unique(idxs)


def _compute_preview_limits(
    before: np.ndarray,
    after: np.ndarray,
    idxs: np.ndarray,
    q_low: float = 1.0,
    q_high: float = 99.0,
) -> tuple[float, float]:
    """
    Compute shared preview limits using BEFORE frames only.
    Non-finite values are ignored.
    """
    del after  # Signature includes `after` intentionally for API stability.
    arr = _validate_frames_thw(before, context="before")
    if idxs.size == 0:
        vals = arr.reshape(-1)
    else:
        vals = arr[idxs].reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(vals, q_low))
    vmax = float(np.percentile(vals, q_high))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        return 0.0, 1.0
    return vmin, vmax


def _sanitize_meta_for_json(value: Any) -> Any:
    """
    Convert metadata tree to JSON-native types and replace non-finite floats with None.
    """
    if isinstance(value, dict):
        return {str(k): _sanitize_meta_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_meta_for_json(v) for v in value]
    if isinstance(value, set):
        return [_sanitize_meta_for_json(v) for v in sorted(value, key=str)]
    if isinstance(value, np.ndarray):
        return _sanitize_meta_for_json(value.tolist())
    if isinstance(value, np.generic):
        return _sanitize_meta_for_json(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    raise TypeError(f"Unsupported metadata type for JSON serialization: {type(value)!r}")


def _build_reoriented_meta(
    meta: dict[str, Any],
    *,
    input_shape: tuple[int, int, int],
    output_shape: tuple[int, int, int],
    rotate_k: int,
    flip_lr: bool,
    target_size: int,
) -> dict[str, Any]:
    meta_out = dict(meta)

    _, h_out, w_out = output_shape
    if h_out == w_out:
        target_size_out = int(h_out)
        was_square = True
    else:
        target_size_out = int(target_size)
        was_square = False

    meta_out["stage"] = STAGE_REORIENTED_RESIZED
    meta_out["rotate_k"] = int(rotate_k)
    meta_out["flip_lr"] = bool(flip_lr)
    meta_out["target_size"] = target_size_out
    meta_out["was_square"] = bool(was_square)
    meta_out["input_shape"] = [int(v) for v in input_shape]
    meta_out["output_shape"] = [int(v) for v in output_shape]
    return _sanitize_meta_for_json(meta_out)


def rotate_frames(frames: np.ndarray, k: int) -> np.ndarray:
    """
    Rotate each frame in a (T,H,W) tensor with np.rot90 around spatial axes.
    """
    arr = _validate_frames_thw(frames, context="rotate_frames")
    return np.rot90(arr, k=int(k), axes=(1, 2))


def flip_frames_lr(frames: np.ndarray) -> np.ndarray:
    """
    Flip each frame in a (T,H,W) tensor left-right (width axis).
    """
    arr = _validate_frames_thw(frames, context="flip_frames_lr")
    return np.flip(arr, axis=2)


def pad_or_crop_to_square(frames: np.ndarray, target_size: int) -> np.ndarray:
    """
    Center pad/crop a (T,H,W) tensor to (T,target_size,target_size).
    No interpolation is performed.
    """
    arr = _validate_frames_thw(frames, context="pad_or_crop_to_square")
    target = int(target_size)
    if target <= 0:
        raise ValueError(f"target_size must be > 0, got {target_size}")

    _, H, W = arr.shape
    out = arr

    if H > target:
        top = (H - target) // 2
        out = out[:, top : top + target, :]
    elif H < target:
        pad_top = (target - H) // 2
        pad_bottom = target - H - pad_top
        out = np.pad(out, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode="constant")

    _, H2, W2 = out.shape
    if W2 > target:
        left = (W2 - target) // 2
        out = out[:, :, left : left + target]
    elif W2 < target:
        pad_left = (target - W2) // 2
        pad_right = target - W2 - pad_left
        out = np.pad(out, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant")

    return out


def reorient_and_resize(
    frames: np.ndarray,
    *,
    rotate_k: int,
    flip_lr: bool,
    target_size: int,
) -> np.ndarray:
    """
    Apply rotate -> optional left-right flip -> center pad/crop to square.
    """
    out = rotate_frames(frames, k=rotate_k)
    if flip_lr:
        out = flip_frames_lr(out)
    out = pad_or_crop_to_square(out, target_size=target_size)
    return out


def save_before_after_preview(
    before: np.ndarray,
    after: np.ndarray,
    out_png_path: str,
    n_frames: int = 6,
) -> None:
    """
    Save a before/after preview grid with shared scaling from BEFORE frames only.
    """
    before_arr = _validate_frames_thw(before, context="save_before_after_preview(before)")
    after_arr = _validate_frames_thw(after, context="save_before_after_preview(after)")
    T = min(before_arr.shape[0], after_arr.shape[0])
    idxs = _select_preview_indices(T=T, n_frames=n_frames)
    if idxs.size == 0:
        return

    vmin, vmax = _compute_preview_limits(before_arr, after_arr, idxs)

    # Keep heavy import local to preview-only path.
    import matplotlib.pyplot as plt

    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = len(idxs)
    fig, axes = plt.subplots(rows, 2, figsize=(7.5, max(2.5, 2.2 * rows)), squeeze=False)

    for row_i, frame_i in enumerate(idxs.tolist()):
        ax_b = axes[row_i, 0]
        ax_a = axes[row_i, 1]

        ax_b.imshow(before_arr[frame_i], cmap="gray", vmin=vmin, vmax=vmax)
        ax_b.set_title(f"Before t={frame_i}")
        ax_b.axis("off")

        ax_a.imshow(after_arr[frame_i], cmap="gray", vmin=vmin, vmax=vmax)
        ax_a.set_title(f"After t={frame_i}")
        ax_a.axis("off")

    fig.suptitle("Reorientation/Resize Preview")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def reorient_baseline_sessions(
    in_npz_paths: list[str],
    out_dir: str | os.PathLike[str],
    *,
    rotate_k: int = -1,
    flip_session_ids: set[str] | None = None,
    target_size: int = 112,
    overwrite: bool = False,
    save_previews: bool = True,
    preview_dir: str | os.PathLike[str] | None = None,
) -> list[str]:
    """
    Run geometry stage for a list of baseline NPZ files and return canonical output paths.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    preview_root = Path(preview_dir) if preview_dir is not None else out_root / "previews"
    flip_ids = flip_session_ids if flip_session_ids is not None else set()

    outputs: list[str] = []
    for in_path_str in in_npz_paths:
        in_path = Path(in_path_str)
        frames, meta = load_stage_npz(str(in_path))
        frames_in = _validate_frames_thw(frames, context=f"{in_path}")
        input_shape = tuple(int(v) for v in frames_in.shape)

        session_id_meta = meta.get("session_id")
        session_id = str(session_id_meta) if session_id_meta not in (None, "") else derive_session_id_from_path(in_path)
        default_flip = session_id in flip_ids

        canonical_out = out_root / f"baseline_{session_id}_{STAGE_REORIENTED_RESIZED}.npz"
        already_reoriented = str(meta.get("stage", "")) == STAGE_REORIENTED_RESIZED

        if already_reoriented:
            rotate_meta = _coerce_int(meta.get("rotate_k"), default=int(rotate_k))
            flip_meta = _coerce_bool(meta.get("flip_lr"), default=default_flip)

            meta_out = _build_reoriented_meta(
                meta,
                input_shape=input_shape,
                output_shape=input_shape,
                rotate_k=rotate_meta,
                flip_lr=flip_meta,
                target_size=int(target_size),
            )
            if overwrite or (not canonical_out.exists()):
                save_stage_npz(str(canonical_out), frames_in, meta_out)

            if save_previews:
                preview_root.mkdir(parents=True, exist_ok=True)
                preview_path = preview_root / f"preview_{session_id}_{STAGE_REORIENTED_RESIZED}.png"
                if overwrite or (not preview_path.exists()):
                    save_before_after_preview(frames_in, frames_in, str(preview_path))

            outputs.append(str(canonical_out))
            continue

        if canonical_out.exists() and not overwrite:
            outputs.append(str(canonical_out))
            continue

        flip_lr = default_flip
        frames_out = reorient_and_resize(
            frames_in,
            rotate_k=int(rotate_k),
            flip_lr=flip_lr,
            target_size=int(target_size),
        )
        output_shape = tuple(int(v) for v in frames_out.shape)

        meta_out = _build_reoriented_meta(
            meta,
            input_shape=input_shape,
            output_shape=output_shape,
            rotate_k=int(rotate_k),
            flip_lr=bool(flip_lr),
            target_size=int(target_size),
        )
        save_stage_npz(str(canonical_out), frames_out, meta_out)

        if save_previews:
            preview_root.mkdir(parents=True, exist_ok=True)
            preview_path = preview_root / f"preview_{session_id}_{STAGE_REORIENTED_RESIZED}.png"
            save_before_after_preview(frames_in, frames_out, str(preview_path))

        outputs.append(str(canonical_out))

    return outputs


# Example sanity usage (do not execute on import):
#   out_paths = reorient_baseline_sessions(
#       in_npz_paths=[...],
#       out_dir=\".../baseline_only_reoriented\",
#       rotate_k=-1,
#       flip_session_ids={\"Se04092020\"},
#       target_size=112,
#   )
