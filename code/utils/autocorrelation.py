import numpy as np
import pandas as pd
from pathlib import Path
from utils import helper_functions as hf

deriv_root = None
eda_root_name = "eda"
acf_qc_filename = "acf_qc_summary.csv"
fdiff_robust_pctl = (2.0, 98.0)
spatial_eps = 1e-8
spatial_min_valid_samples = 8
"""Autocorrelation analysis utilities for fUS data."""


def configure(
    deriv_root_path=None,
    eda_root_name=None,
    acf_qc_filename=None,
    frame_diff_robust_pctl=None,
    spatial_eps=None,
    spatial_min_valid_samples=None,
):
    module_state = globals()
    if deriv_root_path is not None:
        module_state["deriv_root"] = Path(deriv_root_path)
    if eda_root_name is not None:
        module_state["eda_root_name"] = str(eda_root_name)
    if acf_qc_filename is not None:
        module_state["acf_qc_filename"] = str(acf_qc_filename)
    if frame_diff_robust_pctl is not None:
        module_state["fdiff_robust_pctl"] = tuple(
            float(v) for v in frame_diff_robust_pctl
        )
    if spatial_eps is not None:
        module_state["spatial_eps"] = float(spatial_eps)
    if spatial_min_valid_samples is not None:
        module_state["spatial_min_valid_samples"] = int(spatial_min_valid_samples)


def canonical_session_id(value, known_modes):
    s = str(value)
    if s.startswith("baseline_"):
        s = s[len("baseline_"):]
    for mode in sorted(known_modes, key=len, reverse=True):
        suffix = f"_{mode}"
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    return s


def _autodetect_deriv_root():
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        candidate = base / "derivatives" / "preprocessing"
        if candidate.exists():
            return candidate
    return None


def _ensure_deriv_root_or_raise():
    global deriv_root
    if deriv_root is None:
        detected = _autodetect_deriv_root()
        if detected is not None:
            deriv_root = detected
        else:
            raise RuntimeError(
                "autocorrelation.deriv_root is not configured. "
                "Call autocorrelation.configure(deriv_root_path=...) first."
            )
    return deriv_root


def analysis_subject_root(subject, subdir):
    root_base = _ensure_deriv_root_or_raise()
    root = root_base / subject / eda_root_name / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def baseline_npz_path(deriv_root, subject, session_id, mode):
    base_dir = hf.get_baseline_dir(deriv_root, subject, mode)
    if mode == "raw":
        return base_dir / f"baseline_{session_id}.npz"
    return base_dir / f"baseline_{session_id}_{mode}.npz"


def load_saved_session_bundle(deriv_root, subject, session_id, mode):
    fp = baseline_npz_path(deriv_root, subject, session_id, mode)
    if not fp.exists():
        raise FileNotFoundError(f"Missing saved baseline file: {fp}")

    with np.load(fp, allow_pickle=False) as data:
        if "frames" not in data.files:
            raise KeyError(f"'frames' missing in {fp}")
        frames = np.asarray(data["frames"], dtype=np.float32)
        metadata = {k: data[k] for k in data.files if k != "frames"}
    return hf.squeeze_frames(frames).astype(np.float32, copy=False), metadata, fp


def finite_values(a):
    x = np.asarray(a)
    return x[np.isfinite(x)]


def robust_limits(a, pctl=None, symmetric=False, nonnegative=False, eps=1e-8):
    if pctl is None:
        pctl = fdiff_robust_pctl
    x = finite_values(a)
    if x.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(x, pctl)
    lo = float(lo)
    hi = float(hi)
    if symmetric:
        m = max(abs(lo), abs(hi), eps)
        return (-m, m)
    if nonnegative:
        lo = max(0.0, lo)
    if hi <= lo + eps:
        hi = lo + 1.0
    return (lo, hi)

def session_global_signal(frames_3d):
    arr = np.asarray(frames_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got shape {arr.shape}")

    x = np.nanmean(arr, axis=(1, 2)).astype(np.float64)
    finite = np.isfinite(x)
    x = x[finite]

    if x.size < 2:
        return None

    x = x - np.mean(x)
    if not np.isfinite(np.var(x)) or np.var(x) < 1e-12:
        return None
    return x


def normalized_acf(x, max_lag):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return None, None

    corr_full = np.correlate(x, x, mode="full")
    mid = corr_full.size // 2
    corr = corr_full[mid:]

    if corr[0] == 0 or (not np.isfinite(corr[0])):
        return None, None

    L = int(min(max_lag, x.size - 1))
    lags = np.arange(L + 1, dtype=np.int32)
    acf = corr[: L + 1] / corr[0]
    return lags, acf


def load_acf_qc_map_for_subject(subject):
    root_base = _ensure_deriv_root_or_raise()
    qc_path = root_base / subject / acf_qc_filename
    if not qc_path.exists():
        return {}

    try:
        df_qc = pd.read_csv(qc_path)
    except Exception as e:
        print(f"  - warning: could not read QC file for {subject}: {e}")
        return {}

    qc_map = {}
    required_cols = {"session_id", "mode", "pass_acf_qc"}
    if not required_cols.issubset(set(df_qc.columns)):
        print(f"  - warning: QC file missing required columns: {qc_path}")
        return {}

    for _, row in df_qc.iterrows():
        sid = str(row.get("session_id", ""))
        mode = str(row.get("mode", ""))
        passed = bool(row.get("pass_acf_qc", False))
        reason = str(row.get("fail_reason", ""))
        if sid and mode:
            qc_map[(sid, mode)] = (passed, reason)
    return qc_map


def acf_qc_allows_session(qc_map, session_id, mode, require_qc=True):
    if not require_qc:
        return True, ""
    if not qc_map:
        return True, "qc_file_missing_or_unreadable"

    key = (str(session_id), str(mode))
    if key not in qc_map:
        return False, "missing_qc_row"

    passed, reason = qc_map[key]
    if passed:
        return True, ""
    return False, reason if reason else "failed_preprocessing_qc"

"""
Author: Brynn Harris-Shanks, 2026
spatial autocorrelation analysis utilities for fUS data.
"""
def spatial_neighbor_offsets(neighborhood: int):
    n = int(neighborhood)
    if n == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if n == 8:
        return [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
    raise ValueError(f"Unsupported neighborhood={neighborhood!r}; expected 4 or 8")


def safe_temporal_corr_map(A, B, eps=spatial_eps, min_samples=spatial_min_valid_samples):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"A and B must have identical shape; got {A.shape} vs {B.shape}")
    if A.ndim != 3:
        raise ValueError(f"Expected [T,h,w], got shape {A.shape}")

    finite = np.isfinite(A) & np.isfinite(B)
    n_valid = finite.sum(axis=0).astype(np.int32)

    A0 = np.where(finite, A, 0.0)
    B0 = np.where(finite, B, 0.0)

    sum_a = np.sum(A0, axis=0)
    sum_b = np.sum(B0, axis=0)

    mean_a = np.divide(sum_a, n_valid, out=np.zeros_like(sum_a), where=n_valid > 0)
    mean_b = np.divide(sum_b, n_valid, out=np.zeros_like(sum_b), where=n_valid > 0)

    Ac = np.where(finite, A - mean_a[None, :, :], 0.0)
    Bc = np.where(finite, B - mean_b[None, :, :], 0.0)

    cov = np.sum(Ac * Bc, axis=0)
    var_a = np.sum(Ac * Ac, axis=0)
    var_b = np.sum(Bc * Bc, axis=0)
    denom = np.sqrt(var_a * var_b)

    corr = np.full(n_valid.shape, np.nan, dtype=np.float64)
    ok = (n_valid >= int(min_samples)) & np.isfinite(denom) & (denom > float(eps))
    corr[ok] = cov[ok] / denom[ok]

    finite_corr = np.isfinite(corr)
    corr[finite_corr] = np.clip(corr[finite_corr], -1.0, 1.0)
    return corr, n_valid


def seed_patch_mask(shape_hw, seed_center_yx, seed_radius=0):
    shape = tuple(int(v) for v in shape_hw)
    if len(shape) != 2:
        raise ValueError(f"shape_hw must be (H, W), got {shape_hw!r}")
    H, W = shape
    if H <= 0 or W <= 0:
        raise ValueError(f"shape_hw must be positive, got {shape}")

    if seed_center_yx is None or len(seed_center_yx) != 2:
        raise ValueError(
            f"seed_center_yx must be a 2-item sequence (y, x), got {seed_center_yx!r}"
        )
    y = int(seed_center_yx[0])
    x = int(seed_center_yx[1])

    r = int(seed_radius)
    if r < 0:
        raise ValueError(f"seed_radius must be >= 0, got {seed_radius!r}")
    if not (0 <= y < H and 0 <= x < W):
        raise ValueError(
            f"seed center {(y, x)} out of bounds for shape {(H, W)}"
        )

    y0 = max(0, y - r)
    y1 = min(H, y + r + 1)
    x0 = max(0, x - r)
    x1 = min(W, x + r + 1)
    if y1 <= y0 or x1 <= x0:
        raise ValueError(
            f"seed patch is empty for center {(y, x)}, radius={r}, shape={(H, W)}"
        )

    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    if not np.any(mask):
        raise ValueError(
            f"seed patch is empty for center {(y, x)}, radius={r}, shape={(H, W)}"
        )
    return mask


def seed_temporal_corr_map(
    frames_3d,
    seed_center_yx,
    seed_radius=0,
    eps=spatial_eps,
    min_samples=spatial_min_valid_samples,
):
    arr = np.asarray(frames_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"Not enough frames for seed correlation: T={arr.shape[0]}")

    T, H, W = arr.shape
    mask = seed_patch_mask((H, W), seed_center_yx, seed_radius=seed_radius)
    seed_vals = arr[:, mask]
    seed_ts = np.nanmean(seed_vals, axis=1).astype(np.float64)
    finite_seed = np.isfinite(seed_ts)
    n_finite_seed = int(np.sum(finite_seed))
    if n_finite_seed < max(2, int(min_samples)):
        raise ValueError(
            f"Seed trace has insufficient finite samples: {n_finite_seed} "
            f"(min required {max(2, int(min_samples))})"
        )

    seed_var = float(np.var(seed_ts[finite_seed]))
    if (not np.isfinite(seed_var)) or seed_var <= float(eps):
        raise ValueError(
            f"Seed trace is near-constant or invalid (variance={seed_var})"
        )

    seed_3d = np.broadcast_to(seed_ts.reshape(T, 1, 1), arr.shape)
    corr_map, n_valid = safe_temporal_corr_map(
        arr,
        seed_3d,
        eps=eps,
        min_samples=min_samples,
    )
    return corr_map, n_valid, seed_ts, mask


def spatial_autocorr_map(frames_3d, neighborhood=4, eps=spatial_eps, min_samples=spatial_min_valid_samples):
    arr = np.asarray(frames_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"Not enough frames for spatial autocorrelation: T={arr.shape[0]}")

    T, H, W = arr.shape
    sum_map = np.zeros((H, W), dtype=np.float64)
    count_map = np.zeros((H, W), dtype=np.int32)

    for dy, dx in spatial_neighbor_offsets(int(neighborhood)):
        y0 = max(0, -dy)
        y1 = min(H, H - dy)
        x0 = max(0, -dx)
        x1 = min(W, W - dx)

        if (y1 <= y0) or (x1 <= x0):
            continue

        yn0 = y0 + dy
        yn1 = y1 + dy
        xn0 = x0 + dx
        xn1 = x1 + dx

        A = arr[:, y0:y1, x0:x1]
        B = arr[:, yn0:yn1, xn0:xn1]

        corr_local, _ = safe_temporal_corr_map(A, B, eps=eps, min_samples=min_samples)
        finite_local = np.isfinite(corr_local)

        if not np.any(finite_local):
            continue

        sub_sum = sum_map[y0:y1, x0:x1]
        sub_cnt = count_map[y0:y1, x0:x1]
        sub_sum[finite_local] += corr_local[finite_local]
        sub_cnt[finite_local] += 1
        sum_map[y0:y1, x0:x1] = sub_sum
        count_map[y0:y1, x0:x1] = sub_cnt

    sac_map = np.full((H, W), np.nan, dtype=np.float64)
    valid = count_map > 0
    sac_map[valid] = sum_map[valid] / count_map[valid].astype(np.float64)

    finite_map = np.isfinite(sac_map)
    sac_map[finite_map] = np.clip(sac_map[finite_map], -1.0, 1.0)

    global_mean = float(np.nanmean(sac_map)) if np.any(finite_map) else np.nan
    return sac_map, count_map, global_mean


def spatial_limits(map_):
    return (-1.0, 1.0)


def save_spatial_map_csv(path, map_):
    m = np.asarray(map_, dtype=np.float64)
    yy, xx = np.indices(m.shape)
    flat = m.reshape(-1)
    df = pd.DataFrame({
        "y": yy.reshape(-1).astype(np.int32),
        "x": xx.reshape(-1).astype(np.int32),
        "spatial_autocorr": flat,
        "is_finite": np.isfinite(flat).astype(np.int32),
    })
    df.to_csv(path, index=False)


def save_value_map_csv(path, map_, value_col="value"):
    m = np.asarray(map_, dtype=np.float64)
    yy, xx = np.indices(m.shape)
    flat = m.reshape(-1)
    pd.DataFrame({
        "y": yy.reshape(-1).astype(np.int32),
        "x": xx.reshape(-1).astype(np.int32),
        value_col: flat,
    }).to_csv(path, index=False)


# Backward-compatible aliases used in older notebook cells.
_analysis_subject_root = analysis_subject_root
_robust_limits = robust_limits

