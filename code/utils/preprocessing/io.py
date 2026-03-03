from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.io
"""
Author: Brynn Harris-Shanks, 2026
* with adaptations from code by  Leo Sperber, 2025
"""

# Stable stage naming convention used in filenames and meta["stage"].
BASELINE_STAGE_EXTRACTED = "baseline_extracted"
STAGE_REORIENTED_RESIZED = "reoriented_resized"
STAGE_FILTERED = "filtered"
STAGE_STANDARDIZED_MEAN_DIVIDE = "standardized_mean_divide"
STAGE_STANDARDIZED_ZSCORE = "standardized_zscore"
LEGACY_STAGE0_EXTRACTED = "stage0_extracted"

KNOWN_STAGE_SUFFIXES = (
    BASELINE_STAGE_EXTRACTED,
    STAGE_REORIENTED_RESIZED,
    STAGE_FILTERED,
    STAGE_STANDARDIZED_MEAN_DIVIDE,
    STAGE_STANDARDIZED_ZSCORE,
    LEGACY_STAGE0_EXTRACTED,
)

def _repo_root_from_utils() -> Path:
    # io.py -> preprocessing -> utils -> code -> repo_root
    # change to 2 if io.py is moved to code/utils/io.py
    return Path(__file__).resolve().parents[3]


def _tail_after_derivatives_preprocessing(path_obj: Path) -> Optional[Path]:
    parts = path_obj.parts
    lower = [p.lower() for p in parts]
    for i in range(len(parts) - 1):
        if lower[i] == "derivatives" and lower[i + 1] == "preprocessing":
            return Path(*parts[i + 2 :]) if i + 2 < len(parts) else Path()
    return None


def _candidate_baseline_dirs(baseline_dir: str | os.PathLike[str]) -> list[Path]:
    requested = Path(baseline_dir)
    repo_deriv_root = _repo_root_from_utils() / "derivatives" / "preprocessing"
    candidates = [requested]

    tail = _tail_after_derivatives_preprocessing(requested)
    if tail is not None:
        candidates.append(repo_deriv_root / tail)

    deduped: list[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cand)
    return deduped


def _resolve_baseline_dir_with_files(
    baseline_dir: str | os.PathLike[str],
) -> tuple[Path, list[Path]]:
    candidates = _candidate_baseline_dirs(baseline_dir)
    for cand in candidates:
        files = sorted(cand.glob("baseline_*.npz"))
        if files:
            return cand, files
    return candidates[0], []


def _ensure_npz_suffix(out_path: str | os.PathLike[str]) -> Path:
    path = Path(out_path)
    if path.suffix.lower() != ".npz":
        path = Path(f"{path}.npz")
    return path


def _json_coerce_legacy(value: Any) -> Any:
    """
    Coerce legacy NPZ metadata values into JSON-compatible Python types.
    Used only for legacy-schema loading.
    """
    if isinstance(value, dict):
        return {str(k): _json_coerce_legacy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_coerce_legacy(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_coerce_legacy(value.item())
        return [_json_coerce_legacy(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _safe_json_string(value: Any) -> str:
    """
    Best-effort compact JSON string for summary display.
    Falls back to str(value) when not JSON-serializable.
    """
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)


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


def save_stage_npz(out_path: str, frames: np.ndarray, meta: dict) -> None:
    """
    Save stage frames + metadata with a stable NPZ schema.

    Schema:
      - frames: ndarray (float32)
      - meta_json: JSON string

    Notes:
      - This function is intentionally strict about metadata serialization.
      - `meta` must already contain only JSON-native Python types
        (dict/list/str/int/float/bool/None).
      - Callers must convert NumPy values (e.g., np.float32, np.int64, np.ndarray)
        to Python types before calling this function.
      - NaN/Inf values are rejected.
    """
    if not isinstance(meta, dict):
        raise TypeError("meta must be a dict containing only JSON-serializable types.")

    path = _ensure_npz_suffix(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta_out = dict(meta)
    meta_out.setdefault("schema_version", 1)

    try:
        meta_json = json.dumps(meta_out, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "meta must contain only JSON-serializable types (no numpy scalars/arrays, sets, or NaN/Inf)."
        ) from exc

    frames_out = np.asarray(frames).astype(np.float32, copy=False)
    np.savez_compressed(path, frames=frames_out, meta_json=np.array(meta_json))


def load_stage_npz(npz_path: str) -> tuple[np.ndarray, dict]:
    """
    Load frames + metadata from stage NPZ schema.

    Backward compatible with legacy baseline_*.npz files that do not have
    meta_json and instead store metadata in top-level NPZ keys.
    """
    path = Path(npz_path)
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        if "frames" not in files:
            raise KeyError(f"{path}: missing required key 'frames'.")

        frames = np.asarray(data["frames"])

        # New schema
        if "meta_json" in files:
            raw = data["meta_json"]
            if isinstance(raw, np.ndarray):
                if raw.size != 1:
                    raise ValueError(f"{path}: key 'meta_json' must be a scalar JSON string.")
                raw = raw.item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            try:
                meta = json.loads(str(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: failed to decode 'meta_json' as JSON: {exc}") from exc
            if not isinstance(meta, dict):
                raise ValueError(f"{path}: 'meta_json' must decode to a JSON object.")
            meta.setdefault("schema_version", 1)
            return frames, meta

        # Legacy schema fallback
        legacy_meta: dict[str, Any] = {}
        for key in data.files:
            if key == "frames":
                continue
            legacy_meta[key] = _json_coerce_legacy(data[key])
        legacy_meta.setdefault("stage", BASELINE_STAGE_EXTRACTED)
        legacy_meta.setdefault("schema_version", 0)
        if "session_id" in legacy_meta:
            legacy_meta["session_id"] = str(legacy_meta["session_id"])
        return frames, legacy_meta


def summarize_npz_collection(npz_paths: list[str]) -> pd.DataFrame:
    """
    Summarize a collection of stage NPZ files.
    """
    rows: list[dict[str, Any]] = []

    for npz_path in npz_paths:
        p = Path(npz_path)
        row: dict[str, Any] = {
            "path": str(p),
            "filename": p.name,
            "exists": p.exists(),
            "error": "",
            "T": np.nan,
            "H": np.nan,
            "W": np.nan,
            "dtype": "",
            "schema_version": np.nan,
            "session_id": "",
            "stage": "",
            "source_fps": np.nan,
            "n_total_frames": np.nan,
            "n_baseline_frames": np.nan,
            "did_log10": np.nan,
            "log10_eps": np.nan,
            "rotate_k": np.nan,
            "flip_lr": np.nan,
            "target_size": np.nan,
            "did_highpass": np.nan,
            "highpass_params": "",
            "did_pca": np.nan,
            "pca_params": "",
            "did_clip": np.nan,
            "clip_bottom": np.nan,
            "clip_top": np.nan,
            "roi_mode": False,
            "standardize_method": "",
        }

        if not p.exists():
            row["error"] = "file_not_found"
            rows.append(row)
            continue

        try:
            frames, meta = load_stage_npz(str(p))
            arr = np.asarray(frames)
            row["dtype"] = str(arr.dtype)
            if arr.ndim >= 1:
                row["T"] = int(arr.shape[0])
            if arr.ndim >= 2:
                row["H"] = int(arr.shape[1])
            if arr.ndim >= 3:
                row["W"] = int(arr.shape[2])

            row["schema_version"] = meta.get("schema_version", np.nan)
            row["session_id"] = str(meta.get("session_id", ""))
            row["stage"] = str(meta.get("stage", ""))
            row["source_fps"] = meta.get("source_fps", np.nan)
            row["n_total_frames"] = meta.get("n_total_frames", np.nan)
            row["n_baseline_frames"] = meta.get("n_baseline_frames", np.nan)
            row["did_log10"] = meta.get("did_log10", np.nan)
            row["log10_eps"] = meta.get("log10_eps", np.nan)
            row["rotate_k"] = meta.get("rotate_k", np.nan)
            row["flip_lr"] = meta.get("flip_lr", meta.get("flip_lr_applied", np.nan))
            row["target_size"] = meta.get("target_size", np.nan)
            row["did_highpass"] = meta.get("did_highpass", np.nan)
            row["did_pca"] = meta.get("did_pca", np.nan)
            row["did_clip"] = meta.get("did_clip", np.nan)
            row["clip_bottom"] = meta.get("clip_bottom", meta.get("bottom", np.nan))
            row["clip_top"] = meta.get("clip_top", meta.get("top", np.nan))
            row["roi_mode"] = _coerce_bool(meta.get("roi_mode", False), default=False)
            row["standardize_method"] = meta.get(
                "standardize_method", meta.get("standardization", "")
            )

            highpass_params = meta.get("highpass_params", "")
            pca_params = meta.get("pca_params", "")
            row["highpass_params"] = (
                _safe_json_string(highpass_params) if highpass_params not in ("", None) else ""
            )
            row["pca_params"] = (
                _safe_json_string(pca_params) if pca_params not in ("", None) else ""
            )
        except Exception as exc:
            row["error"] = str(exc)

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for col in ("error", "session_id", "stage", "filename"):
        df[col] = df[col].fillna("").astype(str)

    return df.sort_values(
        by=["error", "session_id", "stage", "filename"], kind="stable"
    ).reset_index(drop=True)


def mismatch(images: np.ndarray, labels_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Align image and label sequences by trimming both to the shortest length.
    """
    if images.shape[0] != len(labels_arr):
        min_len = min(images.shape[0], len(labels_arr))
        original_images = images.shape[0]
        original_labels = len(labels_arr)
        images = images[:min_len]
        labels_arr = labels_arr[:min_len]
        print(
            f"  - MISMATCH: Images={original_images}, Labels={original_labels}. "
            f"Shaving to {min_len} frames."
        )
    else:
        print(f"  - Match confirmed: {images.shape[0]} frames and labels.")
    return images, labels_arr


def extract_baseline_frames_from_mat(mat_dict: dict[str, Any]) -> np.ndarray:
    """
    Extract frames from .mat structure and return shape (T, H, W).
    """
    key = "Data" if "Data" in mat_dict else ("Datas" if "Datas" in mat_dict else None)
    if key is None:
        raise KeyError("Neither 'Data' nor 'Datas' found in .mat file.")

    try:
        fus_struct = mat_dict[key]["fus"][0, 0]
        frames = fus_struct["frame"][0, 0]
    except Exception as exc:
        raise KeyError("Failed to access mat[key]['fus'][0,0]['frame'][0,0].") from exc

    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"Expected 3D frames, got shape {frames.shape}.")

    # If likely [H, W, T], transpose to [T, H, W].
    if frames.shape[2] > frames.shape[0] and frames.shape[2] > frames.shape[1]:
        frames = np.transpose(frames, (2, 0, 1))

    return np.asarray(frames, dtype=np.float32)


def extract_fps_from_mat(mat_dict: dict[str, Any]) -> Optional[float]:
    """
    Best-effort extraction of acquisition FPS/frame rate from .mat structure.
    """
    key = "Data" if "Data" in mat_dict else ("Datas" if "Datas" in mat_dict else None)
    if key is None:
        return None

    try:
        fus_struct = mat_dict[key]["fus"][0, 0]
    except Exception:
        return None

    candidate_keys = ("fps", "frame_rate", "framerate", "sampling_rate", "acq_fps")
    fus_fields = fus_struct.dtype.names or ()

    for cand in candidate_keys:
        if cand not in fus_fields:
            continue
        try:
            raw = np.asarray(fus_struct[cand][0, 0]).squeeze()
            if raw.size == 0:
                continue
            val = float(raw)
            if val > 0:
                return val
        except Exception:
            continue

    return None


def load_label_file(label_path: str) -> np.ndarray:
    """
    Load labels from Label_pauses_*.mat file.
    """
    lab = scipy.io.loadmat(label_path)
    if "Datas" not in lab:
        raise KeyError(f"{label_path}: missing top-level key 'Datas'.")
    try:
        labels = lab["Datas"]["Label"][0, 0]
    except Exception as exc:
        raise KeyError(f"{label_path}: failed to access Datas['Label'][0,0].") from exc
    return np.asarray(labels).squeeze()


def extract_and_save_baseline(
    fus_path: str,
    label_path: str,
    output_dir: str,
    *,
    baseline_value: int = -1,
    apply_log10: bool = True,
    log10_eps: float = 1e-6,
    overwrite: bool = False,
) -> Optional[str]:
    """
    Extract baseline frames and save baseline stage NPZ.

    If apply_log10=True, applies log10(frames + eps) before saving.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage = BASELINE_STAGE_EXTRACTED
    date_code = Path(fus_path).stem.replace("Datas_", "")
    out_path = out_dir / f"baseline_{date_code}_{stage}.npz"

    if out_path.exists() and not overwrite:
        print(f"Skipping {date_code} (already exists): {out_path.name}")
        return str(out_path)

    try:
        mat = scipy.io.loadmat(fus_path)
        frames = extract_baseline_frames_from_mat(mat).astype(np.float32, copy=False)
        source_fps = extract_fps_from_mat(mat)

        labels_arr = load_label_file(label_path)
        frames, labels_arr = mismatch(frames, labels_arr)

        baseline_mask = labels_arr == baseline_value
        baseline_indices = np.where(baseline_mask)[0]
        baseline_frames = frames[baseline_mask]

        if baseline_frames.shape[0] == 0:
            print(f"Session {date_code}: no baseline frames found (label == {baseline_value})")
            return None

        if apply_log10:
            eps = float(np.float32(log10_eps))
            baseline_frames = np.log10(
                baseline_frames.astype(np.float32, copy=False) + eps
            ).astype(np.float32, copy=False)
            did_log10 = True
            meta_log10_eps: Optional[float] = eps
        else:
            baseline_frames = baseline_frames.astype(np.float32, copy=False)
            did_log10 = False
            meta_log10_eps = None

        meta: dict[str, Any] = {
            "stage": stage,
            "session_id": date_code,
            "source_fps": source_fps,
            "source_fus_file": os.path.basename(fus_path),
            "source_label_file": os.path.basename(label_path),
            "n_total_frames": int(frames.shape[0]),
            "n_baseline_frames": int(baseline_frames.shape[0]),
            "baseline_value": int(baseline_value),
            "baseline_indices": baseline_indices.astype(np.int64).tolist(),
            "original_shape": [int(v) for v in frames.shape],
            "output_shape": [int(v) for v in baseline_frames.shape],
            "did_log10": did_log10,
            "log10_eps": meta_log10_eps,
        }

        save_stage_npz(str(out_path), baseline_frames, meta)
        print(
            f"Saved {out_path.name}: baseline frames {baseline_frames.shape[0]}/{frames.shape[0]} "
            f"({100.0 * baseline_frames.shape[0] / max(1, frames.shape[0]):.1f}%)"
        )
        return str(out_path)
    except Exception as exc:
        print(f"Error processing {os.path.basename(fus_path)}: {exc}")
        return None


def process_all_baseline_files(
    data_directory: str,
    output_dir: str,
    *,
    overwrite: bool = False,
    apply_log10: bool = True,
    log10_eps: float = 1e-6,
) -> list[str]:
    """
    Extract baseline stage files for all sessions in a source directory.

    apply_log10/log10_eps are passed through to extract_and_save_baseline.
    """
    fus_files = sorted(glob.glob(os.path.join(data_directory, "Datas_*.mat")))
    if len(fus_files) == 0:
        print(f"No Datas_*.mat files found in {data_directory}")
        return []

    print(f"Found {len(fus_files)} fUS files to process")
    saved_paths: list[str] = []

    for fus_path in fus_files:
        date_code = Path(fus_path).stem.replace("Datas_", "")
        label_path = os.path.join(data_directory, f"Label_pauses_{date_code}.mat")
        if not os.path.exists(label_path):
            fallback = sorted(glob.glob(os.path.join(data_directory, f"Label*{date_code}.mat")))
            if fallback:
                label_path = fallback[0]
            else:
                print(f"No label file for {os.path.basename(fus_path)}")
                continue

        out = extract_and_save_baseline(
            fus_path=fus_path,
            label_path=label_path,
            output_dir=output_dir,
            apply_log10=apply_log10,
            log10_eps=log10_eps,
            overwrite=overwrite,
        )
        if out is not None:
            saved_paths.append(out)

    print(f"Extracted baseline from {len(saved_paths)}/{len(fus_files)} sessions")
    return saved_paths


def _derive_session_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("baseline_"):
        stem = stem[len("baseline_") :]
    for stage_suffix in KNOWN_STAGE_SUFFIXES:
        token = f"_{stage_suffix}"
        if stem.endswith(token):
            stem = stem[: -len(token)]
            break
    return stem


def load_baseline_session(baseline_path: str | os.PathLike[str]) -> dict[str, Any]:
    """
    Load one baseline NPZ (stage or legacy) into backward-compatible session dict.
    """
    path = Path(baseline_path)
    frames, meta = load_stage_npz(str(path))
    original_indices = meta.get("baseline_indices", meta.get("original_indices"))
    if original_indices is None:
        original_indices_arr = None
    else:
        try:
            original_indices_arr = np.asarray(original_indices, dtype=np.int64)
        except (TypeError, ValueError):
            original_indices_arr = np.asarray(original_indices)

    session_id = str(meta.get("session_id") or _derive_session_id_from_path(path))
    return {
        "frames": np.asarray(frames),
        "session_id": session_id,
        "original_indices": original_indices_arr,
        "metadata": meta,
    }


def load_all_baseline(baseline_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """
    Load all baseline sessions from a directory.
    """
    resolved_dir, baseline_files = _resolve_baseline_dir_with_files(baseline_dir)
    if len(baseline_files) == 0:
        print(f"WARNING: No baseline_*.npz files found in {resolved_dir}")
        return []

    sessions: list[dict[str, Any]] = []
    for path in baseline_files:
        try:
            sessions.append(load_baseline_session(path))
        except Exception as exc:
            print(f"Error loading {path.name}: {exc}")

    print(f"Loaded {len(sessions)} baseline sessions")
    return sessions
