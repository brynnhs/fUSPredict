import numpy as np

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
            raise ValueError(f"roi_mask shape {roi_mask.shape} != frame spatial shape {frames.shape[1:]}")
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