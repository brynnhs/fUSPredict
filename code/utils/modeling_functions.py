import warnings

import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def iter_windows(dataset, max_items=None):
    """
    Yield (context, target) from dataset as numpy arrays.
    Supports dataset outputs of (x, y) or (x, y, meta).
    """
    n = len(dataset)
    if max_items is not None:
        n = min(n, int(max_items))
    for i in range(n):
        item = dataset[i]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            raise ValueError("Dataset item must be (context, target) or (context, target, meta)")
        yield _to_numpy(x), _to_numpy(y)


def sample_train_frames_for_pca(train_ds, max_frames, seed):
    """
    Sample up to max_frames frames from training windows without loading everything.
    Returns flattened frames of shape [N, V].
    """
    rng = np.random.default_rng(int(seed))
    max_frames = int(max_frames)
    frames = []
    for context, target in iter_windows(train_ds):
        if len(frames) >= max_frames:
            break
        all_frames = np.concatenate([context, target], axis=0)
        n_frames = all_frames.shape[0]
        remaining = max_frames - len(frames)
        if n_frames <= remaining:
            pick = np.arange(n_frames)
        else:
            pick = rng.choice(n_frames, size=remaining, replace=False)
        for idx in pick:
            frame = all_frames[idx, 0]
            frames.append(frame.reshape(-1))
        if len(frames) >= max_frames:
            break
    if len(frames) == 0:
        raise RuntimeError("No frames sampled for PCA.")
    return np.stack(frames, axis=0)


def sample_train_frames_for_pca_per_acq(train_ds, max_frames, seed):
    """
    Sample frames uniformly across acquisitions to avoid sampling bias.
    Returns flattened frames of shape [N, V].
    """
    rng = np.random.default_rng(int(seed))
    max_frames = int(max_frames)
    n_acq = len(train_ds.acq_paths)
    if n_acq <= 0:
        raise RuntimeError("No acquisitions available for PCA sampling.")
    per_acq = max(1, max_frames // n_acq)
    frames = []
    for acq_idx in range(n_acq):
        x = train_ds._load_acquisition(acq_idx)  # [T,1,H,W]
        T = x.shape[0]
        if T <= 0:
            continue
        k = min(per_acq, T)
        idx = rng.choice(T, size=k, replace=False)
        for i in idx:
            frames.append(x[i, 0].reshape(-1))
        if len(frames) >= max_frames:
            break
    if len(frames) == 0:
        raise RuntimeError("No frames sampled for PCA.")
    return np.stack(frames, axis=0)


def compute_frame_metrics(y_true, y_pred, standardize=False, eps=1e-8, decimals=3):
    """
    Compute framewise metrics across pixels.
    If standardize is True, z-score both y_true and y_pred using y_true stats.
    """
    yt = np.asarray(y_true).squeeze()
    yp = np.asarray(y_pred).squeeze()
    if np.issubdtype(yt.dtype, np.integer) or np.issubdtype(yp.dtype, np.integer):
        warnings.warn(
            "compute_frame_metrics received integer inputs; metrics should be computed on float-space arrays.",
            RuntimeWarning,
        )
    yt = yt.astype(np.float32, copy=False)
    yp = yp.astype(np.float32, copy=False)
    yt = yt.reshape(-1)
    yp = yp.reshape(-1)
    if bool(standardize):
        mu = float(yt.mean())
        sigma = float(yt.std())
        if sigma < float(eps):
            sigma = 1.0
        yt = (yt - mu) / sigma
        yp = (yp - mu) / sigma
    err = yp - yt
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))
    denom = np.sum((yt - yt.mean()) ** 2)
    r2 = 1.0 - (np.sum(err ** 2) / denom) if denom > 0 else np.nan
    d = int(decimals)
    return {
        "MSE": float(np.round(mse, d)),
        "RMSE": float(np.round(rmse, d)),
        "MAE": float(np.round(mae, d)),
        "R2": float(np.round(r2, d)),
    }


def evaluate_model_on_dataset(
    predict_fn,
    test_ds,
    max_items=None,
    return_per_window=False,
    standardize=False,
    decimals=3,
):
    """
    Evaluate predict_fn on test_ds. Aggregates mean/std of metrics.
    If standardize is True, each window metric is computed on z-scored targets/predictions.
    """
    per_window = []
    for context, target in iter_windows(test_ds, max_items=max_items):
        pred = predict_fn(context)
        metrics = compute_frame_metrics(
            target[0], pred[0], standardize=standardize, decimals=decimals
        )
        per_window.append(metrics)
    if len(per_window) == 0:
        raise RuntimeError("No windows evaluated.")
    keys = per_window[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in per_window], dtype=float)
        agg[f"{k}_mean"] = float(np.round(vals.mean(), int(decimals)))
        agg[f"{k}_std"] = float(np.round(vals.std(), int(decimals)))
    agg["n_windows"] = len(per_window)
    if return_per_window:
        return agg, per_window
    return agg


def residual_acf_latent(residual_latents, max_lag):
    """
    Compute ACF for each component and a summary of mean abs ACF by lag.
    """
    x = np.asarray(residual_latents)
    if x.ndim != 2:
        raise ValueError("residual_latents must be [T, d]")
    T, d = x.shape
    max_lag = int(max_lag)
    if max_lag < 0 or max_lag >= T:
        raise ValueError("max_lag must be >=0 and < T")
    acf = np.zeros((d, max_lag + 1), dtype=float)
    x_centered = x - x.mean(axis=0, keepdims=True)
    denom = np.sum(x_centered ** 2, axis=0)
    denom = np.where(denom == 0, np.nan, denom)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[:, lag] = 1.0
        else:
            num = np.sum(x_centered[lag:] * x_centered[:-lag], axis=0)
            acf[:, lag] = num / denom
    mean_abs_acf = np.nanmean(np.abs(acf), axis=0)
    return {"acf": acf, "mean_abs_acf_by_lag": mean_abs_acf}


def ljung_box_test(residual_series, lags):
    """
    Run Ljung-Box test on each series if statsmodels is available.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except Exception:
        warnings.warn("statsmodels not available; skipping Ljung-Box test.")
        return None
    x = np.asarray(residual_series)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    results = []
    for i in range(x.shape[1]):
        df = acorr_ljungbox(x[:, i], lags=lags, return_df=True)
        results.append(df)
    return results


def predict_persistence(context, K=1):
    """
    Persistence baseline: x_{t+1} = x_t.
    """
    x = _to_numpy(context)
    last = x[-1]
    pred = np.repeat(last[np.newaxis, ...], int(K), axis=0)
    return pred


def fit_pixel_ar(train_ds, p, ridge_lambda, max_items=None):
    """
    Fit per-pixel AR(p) with ridge.
    Returns dict with A:[H,W,p], b:[H,W].
    """
    p = int(p)
    ridge_lambda = float(ridge_lambda)
    XTX = None
    XTy = None
    H = W = None
    for context, target in iter_windows(train_ds, max_items=max_items):
        if context.shape[0] < p:
            raise ValueError("Context shorter than p.")
        lags = context[-p:, 0]
        y = target[0, 0]
        H, W = y.shape
        V = H * W
        x_lags = lags.reshape(p, V)
        y_flat = y.reshape(V)
        x_aug = np.vstack([np.ones((1, V), dtype=x_lags.dtype), x_lags])
        if XTX is None:
            p1 = p + 1
            XTX = np.zeros((V, p1, p1), dtype=np.float64)
            XTy = np.zeros((V, p1), dtype=np.float64)
        XTX += np.einsum("iv,jv->vij", x_aug, x_aug)
        XTy += np.einsum("iv,v->vi", x_aug, y_flat)
    if XTX is None:
        raise RuntimeError("No training windows for PixelAR.")
    p1 = p + 1
    reg = np.diag([0.0] + [ridge_lambda] * p)
    XTX_reg = XTX + reg[np.newaxis, :, :]
    W_hat = np.linalg.solve(XTX_reg, XTy[..., np.newaxis]).squeeze(-1)
    b = W_hat[:, 0].reshape(H, W).astype(np.float32)
    A = W_hat[:, 1:].reshape(H, W, p).astype(np.float32)
    return {"A": A, "b": b, "p": p}


def predict_pixel_ar(context, params, K=1):
    """
    Predict next frame using fitted per-pixel AR parameters.
    """
    if int(K) != 1:
        raise ValueError("Phase 1 PixelAR supports K=1 only.")
    x = _to_numpy(context)
    p = int(params["p"])
    A = params["A"]
    b = params["b"]
    lags = x[-p:, 0]
    pred = b.copy()
    for i in range(p):
        pred += A[:, :, i] * lags[i]
    pred = pred[np.newaxis, np.newaxis, ...]
    return pred


def _fit_pca(frames_flat, d):
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        raise ImportError("sklearn is required for PCA baselines.") from e
    pca = PCA(n_components=int(d), svd_solver="randomized")
    pca.fit(frames_flat)
    return pca


def _fit_pca_torch(frames_flat, d, device):
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required for GPU PCA baselines.") from e
    x = torch.as_tensor(frames_flat, device=device, dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    q = int(d)
    # pca_lowrank returns V with shape (V, q)
    _, _, v = torch.pca_lowrank(x_centered, q=q, center=False)
    components = v.T.contiguous()  # (q, V)
    return mean.squeeze(0), components


def fit_pca_var(
    train_ds,
    d,
    p,
    ridge_lambda,
    max_pca_frames,
    seed,
    max_items=None,
    use_torch=True,
    device="cuda",
    sample_per_acq=True,
):
    """
    Fit PCA on sampled train frames, then VAR(p) on PCA latents.
    Returns dict with PCA params and VAR weights.
    """
    if sample_per_acq:
        frames_flat = sample_train_frames_for_pca_per_acq(train_ds, max_pca_frames, seed)
    else:
        frames_flat = sample_train_frames_for_pca(train_ds, max_pca_frames, seed)
    if use_torch:
        import torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        try:
            mean_t, components_t = _fit_pca_torch(frames_flat, d, device=device)
            pca_mean = mean_t
            pca_components = components_t
        except RuntimeError as e:
            warnings.warn(f"Torch PCA failed on device '{device}': {e}. Falling back to CPU PCA.")
            use_torch = False
    if not use_torch:
        pca = _fit_pca(frames_flat, d)
        pca_mean = pca.mean_
        pca_components = pca.components_
    p = int(p)
    ridge_lambda = float(ridge_lambda)
    use_torch = bool(use_torch)

    # Always fit VAR on continuous per-acquisition sequences in numpy.
    if hasattr(pca_mean, "detach"):
        pca_mean = pca_mean.detach().cpu().numpy()
    if hasattr(pca_components, "detach"):
        pca_components = pca_components.detach().cpu().numpy()
    mean = np.asarray(pca_mean, dtype=np.float32)
    components = np.asarray(pca_components, dtype=np.float32)

    XTX = None
    XTy = None
    n_used = 0
    for acq_idx in range(len(train_ds.acq_paths)):
        frames = train_ds._load_acquisition(acq_idx)  # [T,1,H,W], float32
        T = frames.shape[0]
        if T <= p:
            continue
        X_flat = frames.reshape(T, -1)
        Z = (X_flat - mean) @ components.T  # [T, d]
        if Z.shape[0] <= p:
            continue
        # Build VAR design matrix for this acquisition.
        Zt = Z[p:]  # targets: z_t for t >= p
        lags = [Z[p - i - 1 : -i - 1] for i in range(p)]  # list of [T-p, d]
        X = np.concatenate(lags, axis=1)  # [T-p, p*d]
        X = np.concatenate([np.ones((X.shape[0], 1), dtype=X.dtype), X], axis=1)
        if XTX is None:
            XTX = X.T @ X
            XTy = X.T @ Zt
        else:
            XTX += X.T @ X
            XTy += X.T @ Zt
        n_used += Zt.shape[0]
        if max_items is not None and n_used >= int(max_items):
            break

    if XTX is None:
        raise RuntimeError("No training sequences for PCA-VAR.")
    reg = np.diag([0.0] + [ridge_lambda] * (XTX.shape[0] - 1))
    W = np.linalg.solve(XTX + reg, XTy)
    var_weights = W.astype(np.float32)
    return {
        "p": p,
        "d": int(d),
        "mean": mean,
        "components": components,
        "var_weights": var_weights,
    }


def predict_pca_var(context, params, K=1):
    """
    Predict next frame using PCA+VAR parameters.
    """
    if int(K) != 1:
        raise ValueError("Phase 1 PCA-VAR supports K=1 only.")
    x = _to_numpy(context)
    p = int(params["p"])
    mean = params["mean"]
    components = params["components"]
    W = params["var_weights"]
    ctx = x[-p:, 0].reshape(p, -1)
    latents = (ctx - mean) @ components.T
    x_feat = latents.reshape(-1)
    x_aug = np.concatenate([np.ones(1, dtype=x_feat.dtype), x_feat], axis=0)
    pred_latent = x_aug @ W
    pred_flat = pred_latent @ components + mean
    H = x.shape[-2]
    Wd = x.shape[-1]
    pred = pred_flat.reshape(H, Wd)
    return pred[np.newaxis, np.newaxis, ...]


def fit_pca_ar_diag(
    train_ds,
    d,
    p,
    ridge_lambda,
    max_pca_frames,
    seed,
    max_items=None,
    sample_per_acq=True,
    use_torch=True,
    device="cuda",
):
    """
    Fit PCA on sampled frames, then independent AR(p) per PCA component.
    Returns dict with PCA params and AR weights per component.
    """
    if sample_per_acq:
        frames_flat = sample_train_frames_for_pca_per_acq(train_ds, max_pca_frames, seed)
    else:
        frames_flat = sample_train_frames_for_pca(train_ds, max_pca_frames, seed)

    use_torch = bool(use_torch)
    if use_torch:
        try:
            import torch
            if str(device).startswith("cuda") and not torch.cuda.is_available():
                warnings.warn("CUDA requested but not available; falling back to CPU PCA.")
                device = "cpu"
            try:
                mean_t, components_t = _fit_pca_torch(frames_flat, d, device=device)
                pca_mean = mean_t
                pca_components = components_t
            except RuntimeError as e:
                warnings.warn(f"Torch PCA failed on device '{device}': {e}. Falling back to CPU PCA.")
                use_torch = False
        except Exception:
            use_torch = False
    if not use_torch:
        pca = _fit_pca(frames_flat, d)
        pca_mean = pca.mean_
        pca_components = pca.components_

    if hasattr(pca_mean, "detach"):
        pca_mean = pca_mean.detach().cpu().numpy()
    if hasattr(pca_components, "detach"):
        pca_components = pca_components.detach().cpu().numpy()
    mean = np.asarray(pca_mean, dtype=np.float32)
    components = np.asarray(pca_components, dtype=np.float32)

    p = int(p)
    ridge_lambda = float(ridge_lambda)

    XTX = None  # [d, p+1, p+1]
    XTy = None  # [d, p+1]
    n_used = 0
    for acq_idx in range(len(train_ds.acq_paths)):
        frames = train_ds._load_acquisition(acq_idx)  # [T,1,H,W]
        T = frames.shape[0]
        if T <= p:
            continue
        X_flat = frames.reshape(T, -1)
        Z = (X_flat - mean) @ components.T  # [T, d]
        if Z.shape[0] <= p:
            continue
        Zt = Z[p:]  # [T-p, d]
        lags = [Z[p - i - 1 : -i - 1] for i in range(p)]  # list of [T-p, d]
        L = np.stack(lags, axis=1)  # [T-p, p, d]
        ones = np.ones((L.shape[0], 1, L.shape[2]), dtype=L.dtype)
        X_aug = np.concatenate([ones, L], axis=1)  # [T-p, p+1, d]
        if XTX is None:
            XTX = np.einsum("tfd,tgd->dfg", X_aug, X_aug)
            XTy = np.einsum("tfd,td->df", X_aug, Zt)
        else:
            XTX += np.einsum("tfd,tgd->dfg", X_aug, X_aug)
            XTy += np.einsum("tfd,td->df", X_aug, Zt)
        n_used += Zt.shape[0]
        if max_items is not None and n_used >= int(max_items):
            break

    if XTX is None:
        raise RuntimeError("No training sequences for PCA-AR.")

    reg = np.diag([0.0] + [ridge_lambda] * p)
    W = np.zeros((XTX.shape[0], p + 1), dtype=np.float32)
    for j in range(XTX.shape[0]):
        W[j] = np.linalg.solve(XTX[j] + reg, XTy[j]).astype(np.float32)

    return {
        "p": p,
        "d": int(d),
        "mean": mean,
        "components": components,
        "ar_weights": W,
    }


def predict_pca_ar_diag(context, params, K=1):
    """
    Predict next frame using PCA + per-component AR(p) parameters.
    """
    if int(K) != 1:
        raise ValueError("Phase 1 PCA-AR supports K=1 only.")
    x = _to_numpy(context)
    p = int(params["p"])
    mean = params["mean"]
    components = params["components"]
    W = params["ar_weights"]  # [d, p+1]

    ctx = x[-p:, 0].reshape(p, -1)
    latents = (ctx - mean) @ components.T  # [p, d]
    # Predict each component independently
    z_pred = W[:, 0].copy()
    for i in range(p):
        z_pred += W[:, i + 1] * latents[-(i + 1)]
    pred_flat = z_pred @ components + mean
    H = x.shape[-2]
    Wd = x.shape[-1]
    pred = pred_flat.reshape(H, Wd)
    return pred[np.newaxis, np.newaxis, ...]


def plot_triplet(gt, pred, residual, title=None):
    """
    Plot GT, prediction, and residual side-by-side.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for ax, img, name in zip(axes, [gt, pred, residual], ["GT", "Pred", "Residual"]):
        ax.imshow(np.asarray(img).squeeze(), cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    if title is not None:
        fig.suptitle(title)
    plt.show()


def save_example_grid(fig, path):
    """
    Save a matplotlib figure to disk.
    """
    fig.savefig(path, dpi=150)
