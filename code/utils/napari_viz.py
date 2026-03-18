from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
from scipy import ndimage as ndi


def check_napari_inline_ready() -> tuple[bool, list[str]]:
    """Return whether inline napari requirements are importable."""
    required_modules = ("napari", "magicgui", "ipywidgets", "jupyter_rfb")
    missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
    return len(missing) == 0, missing


def _to_3d_frames(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected frames shape (T,H,W) or (T,1,H,W), got {arr.shape}")
    return arr


def compute_auto_roi_mask(
    frames: np.ndarray,
    percentile: float = 35.0,
    min_pixels: int = 500,
) -> np.ndarray:
    """
    Compute an automatic ROI mask from temporal mean image.
    This function is in-memory only and never writes to disk.
    """
    arr = _to_3d_frames(frames)
    if not np.isfinite(arr).any():
        return np.ones(arr.shape[1:], dtype=bool)

    mean_map = np.nanmean(arr, axis=0)
    finite_vals = mean_map[np.isfinite(mean_map)]

    if finite_vals.size == 0:
        return np.ones(mean_map.shape, dtype=bool)

    pos_vals = finite_vals[finite_vals > 0]
    ref_vals = pos_vals if pos_vals.size > 0 else finite_vals
    thr = np.percentile(ref_vals, float(percentile))
    mask = np.asarray(mean_map >= thr, dtype=bool)

    labels, nlab = ndi.label(mask)
    if nlab > 0:
        counts = np.bincount(labels.ravel())
        if counts.size > 1:
            counts[0] = 0
            largest = int(np.argmax(counts))
            if counts[largest] >= int(min_pixels):
                mask = labels == largest

    if int(mask.sum()) < max(1, int(min_pixels) // 4):
        return np.ones(mean_map.shape, dtype=bool)

    return mask


def make_patch_rectangle(
    center_yx: tuple[int, int],
    radius: int,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    """
    Return rectangle corners in (y, x) as napari polygon vertices.
    The rectangle is clipped to image bounds and uses half-pixel edges.
    """
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape_hw must be positive, got {shape_hw}")

    r = max(0, int(radius))
    cy = int(np.clip(int(center_yx[0]), 0, h - 1))
    cx = int(np.clip(int(center_yx[1]), 0, w - 1))

    y0 = max(0, cy - r)
    y1 = min(h - 1, cy + r)
    x0 = max(0, cx - r)
    x1 = min(w - 1, cx + r)

    return np.asarray(
        [
            [y0 - 0.5, x0 - 0.5],
            [y0 - 0.5, x1 + 0.5],
            [y1 + 0.5, x1 + 0.5],
            [y1 + 0.5, x0 - 0.5],
        ],
        dtype=np.float32,
    )


def _robust_contrast_limits(
    arr: np.ndarray,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    symmetric: bool = False,
    default: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return default

    if symmetric:
        hi = float(np.percentile(np.abs(vals), p_hi))
        if not np.isfinite(hi) or hi <= 1e-8:
            hi = 1.0
        return -hi, hi

    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return default
    return lo, hi


def _set_layer_attr_if_supported(layer: Any, attr: str, value: Any) -> None:
    """Best-effort layer styling across napari API versions."""
    if not hasattr(layer, attr):
        return
    try:
        setattr(layer, attr, value)
    except Exception:
        return


def _inline_widget_available(viewer: Any) -> bool:
    """
    Detect whether this viewer can render as an inline Jupyter widget.
    If this is False, display(viewer) usually falls back to plain repr text.
    """
    mime_fn = getattr(viewer, "_repr_mimebundle_", None)
    if mime_fn is None:
        return False

    try:
        bundle = mime_fn()
    except Exception:
        return False

    if isinstance(bundle, tuple):
        bundle = bundle[0] if bundle else {}

    if not isinstance(bundle, dict):
        return False

    return "application/vnd.jupyter.widget-view+json" in bundle


def launch_single_session_qc_viewer(
    raw_frames: np.ndarray,
    stdz_frames: np.ndarray,
    mode_name: str,
    session_label: str,
    roi_mask: np.ndarray | None,
    center_yx: tuple[int, int],
    patch_radius: int,
    start_frame: int,
    viewer_mode: str = "inline",
) -> Any:
    """
    Launch napari viewer with raw + standardized stacks and QC overlays.
    Returns napari viewer object.
    """
    raw_arr = _to_3d_frames(raw_frames)
    stdz_arr = _to_3d_frames(stdz_frames)

    if raw_arr.shape != stdz_arr.shape:
        raise ValueError(
            f"raw_frames and stdz_frames must match shape, got {raw_arr.shape} vs {stdz_arr.shape}"
        )

    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "napari is required for launch_single_session_qc_viewer. "
            "Install with `pip install \"napari[all]\" magicgui ipywidgets`."
        ) from exc

    viewer = napari.Viewer(
        title=f"{session_label} | mode={mode_name} | viewer={viewer_mode}",
        show=(viewer_mode != "inline"),
    )
    if viewer_mode == "inline" and not _inline_widget_available(viewer):
        raise RuntimeError(
            "Napari inline widget backend is unavailable in this kernel. "
            "Install/enable `jupyter_rfb` + `ipywidgets`, or use `NAPARI_VIEWER_MODE = \"qt\"`."
        )
    viewer.dims.axis_labels = ("t", "y", "x")

    raw_limits = _robust_contrast_limits(raw_arr, p_lo=1.0, p_hi=99.0, symmetric=False)
    stdz_limits = _robust_contrast_limits(
        stdz_arr,
        p_lo=1.0,
        p_hi=99.0,
        symmetric=True,
        default=(-1.0, 1.0),
    )

    viewer.add_image(
        raw_arr,
        name="raw",
        colormap="gray",
        contrast_limits=raw_limits,
        opacity=1.0,
    )
    viewer.add_image(
        stdz_arr,
        name=f"{mode_name}",
        colormap="bwr",
        contrast_limits=stdz_limits,
        opacity=0.55,
    )

    t, h, w = raw_arr.shape
    cy = int(np.clip(int(center_yx[0]), 0, h - 1))
    cx = int(np.clip(int(center_yx[1]), 0, w - 1))

    if roi_mask is not None:
        roi_arr = np.asarray(roi_mask, dtype=bool)
        if roi_arr.shape == (h, w):
            roi_layer = roi_arr.astype(np.uint8, copy=False)
            try:
                viewer.add_labels(
                    roi_layer,
                    name="roi_mask",
                    opacity=0.20,
                    contour=1,
                )
            except TypeError:
                viewer.add_labels(
                    roi_layer,
                    name="roi_mask",
                    opacity=0.20,
                )

    rect_yx = make_patch_rectangle((cy, cx), int(patch_radius), (h, w))
    rect_tyx = np.stack(
        [
            np.column_stack([np.full(4, i, dtype=np.float32), rect_yx])
            for i in range(t)
        ],
        axis=0,
    )
    patch_layer = viewer.add_shapes(
        rect_tyx,
        shape_type="polygon",
        name="patch",
    )
    _set_layer_attr_if_supported(patch_layer, "edge_color", "yellow")
    _set_layer_attr_if_supported(patch_layer, "face_color", "transparent")
    _set_layer_attr_if_supported(patch_layer, "edge_width", 2)

    points_tyx = np.column_stack(
        [
            np.arange(t, dtype=np.float32),
            np.full(t, cy, dtype=np.float32),
            np.full(t, cx, dtype=np.float32),
        ]
    )
    try:
        points_layer = viewer.add_points(points_tyx, name="center_pixel", size=8)
    except TypeError:
        points_layer = viewer.add_points(points_tyx, name="center_pixel")

    _set_layer_attr_if_supported(points_layer, "edge_color", "yellow")
    _set_layer_attr_if_supported(points_layer, "border_color", "yellow")
    _set_layer_attr_if_supported(points_layer, "face_color", "transparent")
    _set_layer_attr_if_supported(points_layer, "face_color", [0, 0, 0, 0])

    start = int(np.clip(int(start_frame), 0, t - 1))
    viewer.dims.set_current_step(0, start)
    return viewer
