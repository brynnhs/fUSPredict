from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .io import (
    STAGE_FILTERED,
    STAGE_REORIENTED_RESIZED,
    STAGE_STANDARDIZED_MEAN_DIVIDE,
    STAGE_STANDARDIZED_ZSCORE,
    derive_session_id_from_path,
    load_stage_npz,
    save_stage_npz,
)

CONDITION_UNFILTERED = "unfiltered"
CONDITION_FILTERED = "filtered"
_SUPPORTED_INPUT_STAGES = (STAGE_REORIENTED_RESIZED, STAGE_FILTERED)


def _validate_frames_thw(frames: np.ndarray, ctx: str) -> tuple[int, int, int]:
    arr = np.asarray(frames)
    if arr.ndim != 3:
        raise ValueError(f"{ctx}: expected shape (T,H,W), got {arr.shape}")
    t, h, w = arr.shape
    return int(t), int(h), int(w)


def _condition_for_stage(stage: str) -> str:
    if stage == STAGE_REORIENTED_RESIZED:
        return CONDITION_UNFILTERED
    if stage == STAGE_FILTERED:
        return CONDITION_FILTERED
    raise ValueError(
        f"Unsupported input stage {stage!r}; expected one of {_SUPPORTED_INPUT_STAGES!r}."
    )


def _coerce_shape_list(value: Any, key_name: str) -> list[int]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key_name} must be a length-3 sequence [T,H,W], got {value!r}")
    return [int(value[0]), int(value[1]), int(value[2])]


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

    if isinstance(value, float):
        if np.isfinite(value):
            return float(value), 0
        return None, 1

    raise TypeError(f"Unsupported metadata type for JSON serialization: {type(value)!r}")


def standardize_frames_pixelwise(
    frames,
    method="zscore",
    eps=1e-8,
    roi_mask=None,
    floor_percentile=10.0,
    clip_abs=3.0,
):
    """
    standardize raw frames per pixel over time, with robust floors/clipping.

    Parameters
    ----------
    frames : np.ndarray
        Shape (T, H, W).
    method : str
        - "zscore":      (x - mean_t) / std_t per pixel
        - "mean_divide": (x - mean_t) / mean_t per pixel
    eps : float
        Small value for numerical stability.
    roi_mask : np.ndarray or None
        Optional boolean mask of shape (H, W). If provided, floors are estimated
        from ROI-only pixels and non-ROI output is set to 0.
    floor_percentile : float
        Percentile used to compute robust floors for std/|mean| maps. Helps avoid
        exploding values in low-signal pixels.
    clip_abs : float or None
        If provided, clip standardized values to [-clip_abs, +clip_abs].

    Returns
    -------
    np.ndarray
        standardized frames, same shape as input, float32.
    """
    if frames.ndim != 3:
        raise ValueError(f"frames must be (T,H,W), got shape {frames.shape}")

    frames = frames.astype(np.float32, copy=False)
    mean_map = frames.mean(axis=0, keepdims=True)
    std_map = frames.std(axis=0, keepdims=True)

    if roi_mask is not None:
        if roi_mask.shape != frames.shape[1:]:
            raise ValueError(
                f"roi_mask shape {roi_mask.shape} != frame spatial shape {frames.shape[1:]}"
            )
        mask = roi_mask.astype(bool)
    else:
        mask = np.ones(frames.shape[1:], dtype=bool)

    # Robust floors from ROI pixels only.
    std_vals = std_map[0, mask]
    mean_abs_vals = np.abs(mean_map[0, mask])
    std_floor = max(float(np.percentile(std_vals, floor_percentile)), float(eps))
    mean_floor = max(float(np.percentile(mean_abs_vals, floor_percentile)), float(eps))

    if method == "zscore":
        denom = np.maximum(std_map, std_floor)
        norm = (frames - mean_map) / denom
    elif method == "mean_divide":
        denom_mag = np.maximum(np.abs(mean_map), mean_floor)
        denom = np.sign(mean_map) * denom_mag
        denom = np.where(np.abs(denom) < eps, eps, denom)
        norm = (frames - mean_map) / denom
    else:
        raise ValueError("method must be 'zscore' or 'mean_divide'")

    if clip_abs is not None:
        c = float(clip_abs)
        if c > 0:
            norm = np.clip(norm, -c, c)

    if roi_mask is not None:
        norm[:, ~mask] = 0.0

    return norm.astype(np.float32, copy=False)


def build_standardization_meta(
    meta_in: dict,
    *,
    stage: str,
    standardize_method: str,
    eps: float,
    floor_percentile: float,
    clip_abs: float | None,
    roi_mode: bool,
    roi_mask_path: str | None = None,
    input_stage: str | None = None,
    condition: str | None = None,
) -> dict:
    """
    Build JSON-safe metadata for standardized outputs.
    """
    if not isinstance(meta_in, dict):
        raise TypeError("meta_in must be a dict.")

    meta_out: dict[str, Any] = dict(meta_in)
    input_shape = _coerce_shape_list(meta_out.get("input_shape"), "input_shape")
    output_shape = _coerce_shape_list(meta_out.get("output_shape"), "output_shape")

    resolved_input_stage = str(
        input_stage or meta_out.get("input_stage") or meta_out.get("stage") or ""
    )
    if resolved_input_stage not in _SUPPORTED_INPUT_STAGES:
        raise ValueError(
            f"input_stage must be one of {_SUPPORTED_INPUT_STAGES!r}, got {resolved_input_stage!r}"
        )

    resolved_condition = str(condition or meta_out.get("condition") or "")
    if resolved_condition == "":
        resolved_condition = _condition_for_stage(resolved_input_stage)
    expected_condition = _condition_for_stage(resolved_input_stage)
    if resolved_condition != expected_condition:
        raise ValueError(
            f"condition {resolved_condition!r} does not match input_stage {resolved_input_stage!r}"
        )

    meta_out["stage"] = str(stage)
    meta_out["input_stage"] = resolved_input_stage
    meta_out["condition"] = resolved_condition
    meta_out["standardize_method"] = str(standardize_method)
    meta_out["standardize_eps"] = float(eps)
    meta_out["standardize_floor_percentile"] = float(floor_percentile)
    meta_out["standardize_clip_abs"] = float(clip_abs) if clip_abs is not None else ""
    meta_out["roi_mode"] = bool(roi_mode)
    meta_out["roi_mask_path"] = str(roi_mask_path) if roi_mask_path else ""
    meta_out["input_shape"] = input_shape
    meta_out["output_shape"] = output_shape

    meta_sanitized, nonfinite_count = _sanitize_meta_for_json(meta_out)
    if nonfinite_count > 0:
        meta_sanitized["meta_sanitized_nonfinite"] = True
        meta_sanitized["meta_sanitized_nonfinite_count"] = int(nonfinite_count)
    return meta_sanitized


def standardize_one(
    frames: np.ndarray,
    *,
    method: str,
    eps: float,
    roi_mask: np.ndarray | None,
    floor_percentile: float,
    clip_abs: float | None,
) -> np.ndarray:
    """
    Thin wrapper around standardize_frames_pixelwise with shape validation.
    """
    _validate_frames_thw(frames, ctx="standardize_one(frames)")
    out = standardize_frames_pixelwise(
        np.asarray(frames, dtype=np.float32),
        method=method,
        eps=float(eps),
        roi_mask=roi_mask,
        floor_percentile=float(floor_percentile),
        clip_abs=None if clip_abs is None else float(clip_abs),
    )
    return np.asarray(out, dtype=np.float32)


def standardize_stage_sessions(
    in_npz_paths: list[str],
    out_dir: str | os.PathLike[str],
    *,
    eps: float = 1e-8,
    floor_percentile: float = 10.0,
    clip_abs: float | None = 3.0,
    roi_mode: bool = False,
    roi_mask_by_session: dict[str, np.ndarray] | None = None,
    roi_mask_path_by_session: dict[str, str] | None = None,
    allowed_stages: tuple[str, ...] = (STAGE_REORIENTED_RESIZED, STAGE_FILTERED),
    overwrite: bool = False,
) -> dict[str, list[str]]:
    """
    Standardize sessions from supported preprocessing stages into condition-aware outputs.
    """
    if len(allowed_stages) == 0:
        raise ValueError("allowed_stages must not be empty.")

    allowed_stages_norm = tuple(str(s) for s in allowed_stages)
    invalid_allowed = [s for s in allowed_stages_norm if s not in _SUPPORTED_INPUT_STAGES]
    if invalid_allowed:
        raise ValueError(
            f"allowed_stages contains unsupported values {invalid_allowed!r}; "
            f"supported={_SUPPORTED_INPUT_STAGES!r}"
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    n_inputs = len(in_npz_paths)
    outputs: dict[str, list[str]] = {
        "unfiltered_mean_divide": [""] * n_inputs,
        "unfiltered_zscore": [""] * n_inputs,
        "filtered_mean_divide": [""] * n_inputs,
        "filtered_zscore": [""] * n_inputs,
    }

    for idx, in_path_str in enumerate(in_npz_paths):
        in_path = Path(in_path_str)
        frames, meta = load_stage_npz(str(in_path))
        frames_in = np.asarray(frames, dtype=np.float32)
        t, h, w = _validate_frames_thw(frames_in, ctx=str(in_path))

        stage_in = str(meta.get("stage", ""))
        if stage_in not in allowed_stages_norm:
            raise ValueError(
                f"{in_path}: expected input stage in {allowed_stages_norm!r}, got {stage_in!r}"
            )

        condition = _condition_for_stage(stage_in)
        session_meta = meta.get("session_id")
        session_id = (
            str(session_meta)
            if session_meta not in (None, "")
            else derive_session_id_from_path(in_path)
        )

        roi_mask = None
        roi_mask_path = ""
        if roi_mode:
            if roi_mask_by_session is None or session_id not in roi_mask_by_session:
                raise ValueError(
                    f"roi_mode=True requires roi_mask_by_session[{session_id!r}]"
                )
            roi_mask_arr = np.asarray(roi_mask_by_session[session_id])
            if roi_mask_arr.shape != (h, w):
                raise ValueError(
                    f"session {session_id!r}: roi mask shape {roi_mask_arr.shape} "
                    f"!= {(h, w)}"
                )
            roi_mask = roi_mask_arr.astype(bool, copy=False)
            if (
                roi_mask_path_by_session is not None
                and session_id in roi_mask_path_by_session
            ):
                roi_mask_path = str(roi_mask_path_by_session[session_id])

        mean_path = (
            out_root
            / f"baseline_{session_id}_{condition}_{STAGE_STANDARDIZED_MEAN_DIVIDE}.npz"
        )
        zscore_path = (
            out_root / f"baseline_{session_id}_{condition}_{STAGE_STANDARDIZED_ZSCORE}.npz"
        )

        need_mean = overwrite or (not mean_path.exists())
        need_zscore = overwrite or (not zscore_path.exists())

        meta_base: dict[str, Any] = dict(meta)
        meta_base["session_id"] = session_id
        meta_base["input_shape"] = [int(t), int(h), int(w)]
        meta_base["output_shape"] = [int(t), int(h), int(w)]
        meta_base["input_stage"] = stage_in
        meta_base["condition"] = condition

        if need_mean:
            mean_frames = standardize_one(
                frames_in,
                method="mean_divide",
                eps=eps,
                roi_mask=roi_mask,
                floor_percentile=floor_percentile,
                clip_abs=clip_abs,
            )
            mean_meta = build_standardization_meta(
                meta_base,
                stage=STAGE_STANDARDIZED_MEAN_DIVIDE,
                standardize_method="mean_divide",
                eps=eps,
                floor_percentile=floor_percentile,
                clip_abs=clip_abs,
                roi_mode=roi_mode,
                roi_mask_path=roi_mask_path,
                input_stage=stage_in,
                condition=condition,
            )
            save_stage_npz(str(mean_path), mean_frames, mean_meta)

        if need_zscore:
            zscore_frames = standardize_one(
                frames_in,
                method="zscore",
                eps=eps,
                roi_mask=roi_mask,
                floor_percentile=floor_percentile,
                clip_abs=clip_abs,
            )
            zscore_meta = build_standardization_meta(
                meta_base,
                stage=STAGE_STANDARDIZED_ZSCORE,
                standardize_method="zscore",
                eps=eps,
                floor_percentile=floor_percentile,
                clip_abs=clip_abs,
                roi_mode=roi_mode,
                roi_mask_path=roi_mask_path,
                input_stage=stage_in,
                condition=condition,
            )
            save_stage_npz(str(zscore_path), zscore_frames, zscore_meta)

        if condition == CONDITION_UNFILTERED:
            outputs["unfiltered_mean_divide"][idx] = str(mean_path)
            outputs["unfiltered_zscore"][idx] = str(zscore_path)
        elif condition == CONDITION_FILTERED:
            outputs["filtered_mean_divide"][idx] = str(mean_path)
            outputs["filtered_zscore"][idx] = str(zscore_path)

    return outputs


def standardize_filtered_sessions(
    in_npz_paths: list[str],
    out_dir: str | os.PathLike[str],
    *,
    eps: float = 1e-8,
    floor_percentile: float = 10.0,
    clip_abs: float | None = 3.0,
    roi_mode: bool = False,
    roi_mask_by_session: dict[str, np.ndarray] | None = None,
    roi_mask_path_by_session: dict[str, str] | None = None,
    overwrite: bool = False,
) -> dict[str, list[str]]:
    """
    Backward-compatible filtered-only wrapper with input-aligned outputs.
    """
    out = standardize_stage_sessions(
        in_npz_paths,
        out_dir=out_dir,
        eps=eps,
        floor_percentile=floor_percentile,
        clip_abs=clip_abs,
        roi_mode=roi_mode,
        roi_mask_by_session=roi_mask_by_session,
        roi_mask_path_by_session=roi_mask_path_by_session,
        allowed_stages=(STAGE_FILTERED,),
        overwrite=overwrite,
    )
    return {
        "standardized_mean_divide": out["filtered_mean_divide"],
        "standardized_zscore": out["filtered_zscore"],
    }
