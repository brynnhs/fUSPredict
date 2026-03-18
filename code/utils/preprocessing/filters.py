from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .io import (
    STAGE_FILTERED,
    derive_session_id_from_path,
    load_stage_npz,
    save_stage_npz,
)


def _validate_frames_thw(frames: np.ndarray, ctx: str) -> tuple[int, int, int]:
    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError(f"{ctx}: expected shape (T,H,W), got {arr.shape}")
    t, h, w = arr.shape
    return int(t), int(h), int(w)


def _is_finite_number(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, (int, np.integer)):
        return True
    if isinstance(x, (float, np.floating)):
        return bool(np.isfinite(float(x)))
    return False


def _sanitize_meta_for_json(value: Any) -> tuple[Any, int]:
    """
    Convert metadata to JSON-native Python types.
    Non-finite floats become None and contribute to a non-finite counter.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        total = 0
        for k, v in value.items():
            vv, cc = _sanitize_meta_for_json(v)
            out[str(k)] = vv
            total += cc
        return out, total

    if isinstance(value, (list, tuple)):
        out_list: list[Any] = []
        total = 0
        for v in value:
            vv, cc = _sanitize_meta_for_json(v)
            out_list.append(vv)
            total += cc
        return out_list, total

    if isinstance(value, set):
        return _sanitize_meta_for_json(sorted(value, key=str))

    if isinstance(value, np.ndarray):
        return _sanitize_meta_for_json(value.tolist())

    if isinstance(value, np.generic):
        return _sanitize_meta_for_json(value.item())

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace"), 0

    if isinstance(value, Path):
        return str(value), 0

    if value is None or isinstance(value, (str, bool, int)):
        return value, 0

    if isinstance(value, (float, np.floating)):
        val = float(value)
        if np.isfinite(val):
            return val, 0
        return None, 1

    raise TypeError(f"Unsupported metadata type for JSON serialization: {type(value)!r}")


def high_pass_filter(
    frames: np.ndarray,
    fps: float,
    cutoff_hz: float,
    order: int = 3,
) -> np.ndarray:
    """
    Temporal high-pass filter over axis=0 for (T,H,W) frames.
    """
    t, _, _ = _validate_frames_thw(frames, ctx="high_pass_filter(frames)")
    arr = np.asarray(frames, dtype=np.float32)

    if not _is_finite_number(fps) or float(fps) <= 0.0:
        raise ValueError(f"fps must be finite and > 0, got {fps!r}")
    if not _is_finite_number(cutoff_hz) or float(cutoff_hz) <= 0.0:
        raise ValueError(f"cutoff_hz must be finite and > 0, got {cutoff_hz!r}")
    if not isinstance(order, (int, np.integer)) or int(order) < 1:
        raise ValueError(f"order must be >= 1, got {order!r}")

    nyquist = float(fps) / 2.0
    wn = float(cutoff_hz) / nyquist
    if not (0.0 < wn < 1.0):
        raise ValueError(
            f"cutoff_hz must satisfy 0 < cutoff_hz < nyquist ({nyquist}), got {cutoff_hz!r}"
        )

    from scipy import signal

    b, a = signal.butter(int(order), wn, btype="high", analog=False)
    padlen = 3 * max(len(a), len(b))

    if t <= padlen:
        warnings.warn(
            f"high_pass_filter: sequence length T={t} is too short for filtfilt "
            f"(padlen={padlen}); falling back to lfilter.",
            stacklevel=2,
        )
        out = signal.lfilter(b, a, arr, axis=0)
        return np.asarray(out, dtype=np.float32)

    try:
        out = signal.filtfilt(b, a, arr, axis=0)
    except ValueError as exc:
        warnings.warn(
            f"high_pass_filter: filtfilt failed ({exc}); falling back to lfilter.",
            stacklevel=2,
        )
        out = signal.lfilter(b, a, arr, axis=0)

    return np.asarray(out, dtype=np.float32)


def pca_denoise(
    frames: np.ndarray,
    *,
    var_keep: float = 0.95,
    n_components: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    PCA denoise where time (T) is sample axis and flattened pixels are features.
    """
    t, h, w = _validate_frames_thw(frames, ctx="pca_denoise(frames)")
    arr = np.asarray(frames, dtype=np.float32)

    if n_components is not None:
        if not isinstance(n_components, (int, np.integer)) or int(n_components) < 1:
            raise ValueError(f"n_components must be a positive int, got {n_components!r}")
    else:
        if not _is_finite_number(var_keep) or not (0.0 < float(var_keep) <= 1.0):
            raise ValueError(f"var_keep must be in (0, 1], got {var_keep!r}")

    available = min(t, h * w)
    if available <= 0:
        meta_empty: dict[str, Any] = {
            "method": "pca_denoise",
            "n_components": 0,
        }
        if n_components is None:
            meta_empty["var_keep"] = float(var_keep)
        return arr.astype(np.float32, copy=False), meta_empty

    from sklearn.decomposition import PCA

    x = arr.reshape(t, h * w)
    pca = PCA()
    pca.fit(x)

    if n_components is None:
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        chosen = int(np.searchsorted(cumvar, float(var_keep), side="left") + 1)
        chosen = max(1, min(chosen, available))
    else:
        chosen = max(1, min(int(n_components), available))

    x_trans = pca.transform(x)
    x_subset = x_trans[:, :chosen]
    components_subset = pca.components_[:chosen, :]
    x_rec = x_subset @ components_subset + pca.mean_

    out = x_rec.reshape(t, h, w).astype(np.float32, copy=False)

    pca_meta: dict[str, Any] = {
        "method": "pca_denoise",
        "n_components": int(chosen),
    }
    if n_components is None:
        pca_meta["var_keep"] = float(var_keep)
    if pca.explained_variance_ratio_.size > 0:
        pca_meta["explained_variance_ratio_sum"] = float(
            np.sum(pca.explained_variance_ratio_[:chosen])
        )

    return out, pca_meta


def percentile_clip(
    frames: np.ndarray,
    bottom: float = 1.0,
    top: float = 99.0,
) -> np.ndarray:
    """
    Global percentile clipping over all values in (T,H,W).
    """
    _validate_frames_thw(frames, ctx="percentile_clip(frames)")
    arr = np.asarray(frames, dtype=np.float32)

    if not _is_finite_number(bottom) or not _is_finite_number(top):
        raise ValueError("bottom/top must be finite numbers.")

    bottom_f = float(bottom)
    top_f = float(top)
    if not (0.0 <= bottom_f < top_f <= 100.0):
        raise ValueError(f"Expected 0 <= bottom < top <= 100, got {bottom_f}, {top_f}")

    if arr.size == 0:
        return arr.astype(np.float32, copy=False)

    lo = float(np.percentile(arr, bottom_f))
    hi = float(np.percentile(arr, top_f))
    out = np.clip(arr, lo, hi)
    return np.asarray(out, dtype=np.float32)


def apply_optional_filters(
    frames: np.ndarray,
    meta: dict[str, Any],
    *,
    enable_highpass: bool = False,
    highpass_cutoff_hz: float = 0.0,
    highpass_order: int = 3,
    enable_pca: bool = False,
    pca_var_keep: float = 0.95,
    pca_n_components: int | None = None,
    enable_clip: bool = False,
    clip_bottom: float = 1.0,
    clip_top: float = 99.0,
    fps_fallback: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply enabled filters in fixed order:
      1) high-pass, 2) PCA denoise, 3) percentile clip.
    """
    if not isinstance(meta, dict):
        raise TypeError("meta must be a dict.")

    arr = np.asarray(frames, dtype=np.float32)
    t, h, w = _validate_frames_thw(arr, ctx="apply_optional_filters(frames)")

    session_id = str(meta.get("session_id", ""))
    ctx = f"session={session_id}" if session_id else "session=<unknown>"

    meta_out: dict[str, Any] = dict(meta)
    frames_out = arr

    if enable_highpass:
        fps_meta = meta.get("source_fps")
        fps: float | None = None
        if _is_finite_number(fps_meta):
            fps = float(fps_meta)
        elif _is_finite_number(fps_fallback):
            fps = float(fps_fallback)

        if fps is None:
            raise ValueError(
                f"{ctx}: high-pass enabled but no finite fps provided in "
                f"meta['source_fps'] or fps_fallback."
            )

        frames_out = high_pass_filter(
            frames_out,
            fps=fps,
            cutoff_hz=float(highpass_cutoff_hz),
            order=int(highpass_order),
        )
        did_highpass = True
        highpass_params: dict[str, Any] | str = {
            "cutoff_hz": float(highpass_cutoff_hz),
            "order": int(highpass_order),
            "fps": float(fps),
        }
    else:
        did_highpass = False
        highpass_params = ""

    if enable_pca:
        frames_out, pca_meta = pca_denoise(
            frames_out,
            var_keep=float(pca_var_keep),
            n_components=pca_n_components,
        )
        did_pca = True
        pca_params: dict[str, Any] | str = pca_meta
    else:
        did_pca = False
        pca_params = ""

    if enable_clip:
        frames_out = percentile_clip(
            frames_out,
            bottom=float(clip_bottom),
            top=float(clip_top),
        )
        did_clip = True
    else:
        did_clip = False

    meta_out["stage"] = STAGE_FILTERED
    meta_out["did_highpass"] = bool(did_highpass)
    meta_out["highpass_params"] = highpass_params
    meta_out["did_pca"] = bool(did_pca)
    meta_out["pca_params"] = pca_params
    meta_out["did_clip"] = bool(did_clip)
    if did_clip:
        meta_out["clip_bottom"] = float(clip_bottom)
        meta_out["clip_top"] = float(clip_top)
    else:
        meta_out.pop("clip_bottom", None)
        meta_out.pop("clip_top", None)

    meta_out["input_shape"] = [int(t), int(h), int(w)]
    meta_out["output_shape"] = [int(v) for v in frames_out.shape]

    meta_sanitized, nonfinite_count = _sanitize_meta_for_json(meta_out)
    if nonfinite_count > 0:
        meta_sanitized["meta_sanitized_nonfinite"] = True
        meta_sanitized["meta_sanitized_nonfinite_count"] = int(nonfinite_count)

    return np.asarray(frames_out, dtype=np.float32), meta_sanitized


def filter_reoriented_sessions(
    in_npz_paths: list[str],
    out_dir: str | os.PathLike[str],
    *,
    enable_highpass: bool = False,
    highpass_cutoff_hz: float = 0.0,
    highpass_order: int = 3,
    enable_pca: bool = False,
    pca_var_keep: float = 0.95,
    pca_n_components: int | None = None,
    enable_clip: bool = False,
    clip_bottom: float = 1.0,
    clip_top: float = 99.0,
    fps_fallback: float | None = None,
    overwrite: bool = False,
) -> list[str]:
    """
    Run filtering stage and save canonical filtered files in output directory.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []
    for in_path_str in in_npz_paths:
        in_path = Path(in_path_str)
        frames, meta = load_stage_npz(str(in_path))
        t, h, w = _validate_frames_thw(frames, ctx=str(in_path))
        frames_in = np.asarray(frames, dtype=np.float32)

        session_meta = meta.get("session_id")
        session_id = (
            str(session_meta)
            if session_meta not in (None, "")
            else derive_session_id_from_path(in_path)
        )
        canonical_out = out_root / f"baseline_{session_id}_{STAGE_FILTERED}.npz"

        is_already_filtered = str(meta.get("stage", "")) == STAGE_FILTERED
        if is_already_filtered:
            meta_out: dict[str, Any] = dict(meta)
            meta_out["stage"] = STAGE_FILTERED
            meta_out["session_id"] = session_id
            meta_out["input_shape"] = [int(t), int(h), int(w)]
            meta_out["output_shape"] = [int(t), int(h), int(w)]

            meta_sanitized, nonfinite_count = _sanitize_meta_for_json(meta_out)
            if nonfinite_count > 0:
                meta_sanitized["meta_sanitized_nonfinite"] = True
                meta_sanitized["meta_sanitized_nonfinite_count"] = int(nonfinite_count)

            if overwrite or (not canonical_out.exists()):
                save_stage_npz(str(canonical_out), frames_in, meta_sanitized)
            outputs.append(str(canonical_out))
            continue

        if canonical_out.exists() and not overwrite:
            outputs.append(str(canonical_out))
            continue

        meta_for_filter = dict(meta)
        meta_for_filter.setdefault("session_id", session_id)

        frames_out, meta_out = apply_optional_filters(
            frames_in,
            meta_for_filter,
            enable_highpass=enable_highpass,
            highpass_cutoff_hz=highpass_cutoff_hz,
            highpass_order=highpass_order,
            enable_pca=enable_pca,
            pca_var_keep=pca_var_keep,
            pca_n_components=pca_n_components,
            enable_clip=enable_clip,
            clip_bottom=clip_bottom,
            clip_top=clip_top,
            fps_fallback=fps_fallback,
        )
        save_stage_npz(str(canonical_out), frames_out, meta_out)
        outputs.append(str(canonical_out))

    return outputs
