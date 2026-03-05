from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class _PreparedInput:
    x_df: pd.DataFrame
    labels: pd.Series | None
    source_type: str
    input_shape: list[int]
    feature_names: list[str]
    sample_index: pd.Index


def _resolve_feature_names(
    data: pd.DataFrame | np.ndarray,
    features: list[str] | tuple[str, ...] | None,
    label_col: str | None,
) -> list[str]:
    if isinstance(data, pd.DataFrame):
        if features is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if label_col is not None:
                numeric_cols = [c for c in numeric_cols if c != str(label_col)]
            if not numeric_cols:
                raise ValueError("No numeric columns available for PCA.")
            return numeric_cols

        feats = [str(c) for c in features]
        missing = [c for c in feats if c not in data.columns]
        if missing:
            raise ValueError(f"Requested features not found in DataFrame: {missing}")
        return feats

    arr = np.asarray(data)
    if arr.ndim == 2:
        n_features = int(arr.shape[1])
        if features is None:
            return [f"f{i}" for i in range(n_features)]
        feats = [str(c) for c in features]
        if len(feats) != n_features:
            raise ValueError(
                f"features length ({len(feats)}) must match ndarray feature count ({n_features})."
            )
        return feats

    if arr.ndim == 3:
        _, h, w = arr.shape
        n_features = int(h * w)
        if features is None:
            return [f"pix_r{r}_c{c}" for r in range(int(h)) for c in range(int(w))]
        feats = [str(c) for c in features]
        if len(feats) != n_features:
            raise ValueError(
                f"features length ({len(feats)}) must match flattened feature count ({n_features})."
            )
        return feats

    raise ValueError(f"Unsupported ndarray shape {arr.shape}; expected 2D or 3D.")


def _prepare_input_matrix(
    data: pd.DataFrame | np.ndarray,
    *,
    features: list[str] | tuple[str, ...] | None = None,
    label_col: str | None = None,
) -> _PreparedInput:
    feat_names = _resolve_feature_names(data, features, label_col)

    if isinstance(data, pd.DataFrame):
        x_df = data.loc[:, feat_names].copy()
        for col in x_df.columns:
            if not pd.api.types.is_numeric_dtype(x_df[col]):
                try:
                    x_df[col] = pd.to_numeric(x_df[col], errors="raise")
                except Exception as exc:  # pragma: no cover - exact parser error is backend-dependent
                    raise ValueError(
                        f"Feature {col!r} is non-numeric and could not be converted for PCA."
                    ) from exc
        labels = data[label_col].copy() if (label_col is not None and label_col in data.columns) else None
        source_type = "dataframe"
        input_shape = [int(v) for v in data.shape]
        sample_index = data.index
    else:
        arr = np.asarray(data)
        if arr.ndim == 2:
            x = np.asarray(arr, dtype=np.float64)
            source_type = "ndarray_2d"
            input_shape = [int(v) for v in arr.shape]
        elif arr.ndim == 3:
            t, h, w = arr.shape
            x = np.asarray(arr.reshape(t, h * w), dtype=np.float64)
            source_type = "ndarray_3d"
            input_shape = [int(v) for v in arr.shape]
        else:
            raise ValueError(f"Unsupported ndarray shape {arr.shape}; expected 2D or 3D.")
        x_df = pd.DataFrame(x, columns=feat_names)
        labels = None
        sample_index = x_df.index

    if x_df.shape[0] < 2:
        raise ValueError(f"PCA requires at least 2 samples, got {x_df.shape[0]}.")
    if x_df.shape[1] < 2:
        raise ValueError(f"PCA requires at least 2 features, got {x_df.shape[1]}.")

    return _PreparedInput(
        x_df=x_df,
        labels=labels,
        source_type=source_type,
        input_shape=input_shape,
        feature_names=feat_names,
        sample_index=sample_index,
    )


def _apply_missing_policy(
    x_df: pd.DataFrame,
    *,
    labels: pd.Series | None = None,
    missing: str = "median",
) -> tuple[pd.DataFrame, pd.Series | None, dict[str, Any]]:
    policy = str(missing).strip().lower()
    if policy not in {"median", "drop_rows", "error"}:
        raise ValueError(f"missing must be one of ['median', 'drop_rows', 'error'], got {missing!r}")

    x = x_df.copy()
    y = labels.copy() if labels is not None else None

    nan_before = int(x.isna().sum().sum())
    rows_before = int(x.shape[0])
    cols_before = int(x.shape[1])

    meta: dict[str, Any] = {
        "missing_policy": policy,
        "nan_before": nan_before,
        "rows_before": rows_before,
        "cols_before": cols_before,
        "rows_dropped": 0,
        "nans_imputed": 0,
        "all_nan_features_replaced_with_zero": 0,
    }

    if policy == "error":
        if nan_before > 0:
            raise ValueError("Input contains NaNs and missing='error' was requested.")
        return x, y, meta

    if policy == "drop_rows":
        keep_mask = ~x.isna().any(axis=1)
        x = x.loc[keep_mask].copy()
        if y is not None:
            y = y.loc[keep_mask].copy()
        meta["rows_dropped"] = int(rows_before - x.shape[0])
        if x.shape[0] < 2:
            raise ValueError("After dropping NaN rows, fewer than 2 samples remain.")
        return x, y, meta

    medians = x.median(axis=0, skipna=True)
    all_nan_cols = medians.index[medians.isna()].tolist()
    if all_nan_cols:
        medians.loc[all_nan_cols] = 0.0
    x = x.fillna(medians)
    meta["nans_imputed"] = nan_before
    meta["all_nan_features_replaced_with_zero"] = int(len(all_nan_cols))
    if int(x.isna().sum().sum()) > 0:
        raise ValueError("NaNs remain after median imputation; check input dtype/content.")
    return x, y, meta


def _apply_scaling(
    x: np.ndarray,
    *,
    scale: bool = True,
) -> tuple[np.ndarray, StandardScaler | None, dict[str, Any]]:
    arr = np.asarray(x, dtype=np.float64)
    if not bool(scale):
        return arr, None, {"scaled": False}

    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    arr_scaled = scaler.fit_transform(arr)
    return arr_scaled, scaler, {"scaled": True}


def _fit_pca_model(
    x: np.ndarray,
    *,
    mode: str = "var_keep",
    var_keep: float = 0.95,
    n_components: int | None = None,
) -> dict[str, Any]:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"var_keep", "n_components"}:
        raise ValueError(f"mode must be 'var_keep' or 'n_components', got {mode!r}")

    pca = PCA(svd_solver="full")
    scores = pca.fit_transform(x)
    evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    evr_cum = np.cumsum(evr)
    available = int(evr.shape[0])
    if available == 0:
        raise ValueError("PCA returned zero components.")

    if mode_norm == "n_components":
        if n_components is None:
            raise ValueError("n_components must be set when mode='n_components'.")
        k = max(1, min(int(n_components), available))
    else:
        var_keep_f = float(var_keep)
        if not (0.0 < var_keep_f <= 1.0):
            raise ValueError(f"var_keep must be in (0,1], got {var_keep!r}")
        k = int(np.searchsorted(evr_cum, var_keep_f, side="left") + 1)
        k = max(1, min(k, available))

    return {
        "pca": pca,
        "scores": scores,
        "evr": evr,
        "evr_cum": evr_cum,
        "available_components": available,
        "k_chosen": int(k),
    }


def plot_scree_cumulative(
    explained_variance_ratio: np.ndarray,
    explained_variance_ratio_cum: np.ndarray,
    *,
    k_chosen: int,
    title: str = "PCA scree and cumulative EVR",
) -> plt.Figure:
    evr = np.asarray(explained_variance_ratio, dtype=float)
    cum = np.asarray(explained_variance_ratio_cum, dtype=float)
    x = np.arange(1, evr.size + 1, dtype=int)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(x, evr, marker="o", lw=1, label="EVR")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.axvline(int(k_chosen), color="tab:red", ls="--", lw=1, label=f"k={int(k_chosen)}")

    ax2 = ax1.twinx()
    ax2.plot(x, cum, color="tab:orange", lw=2, label="Cumulative EVR")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0.0, 1.01)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_scores_scatter(
    scores_df: pd.DataFrame,
    *,
    pc_x: int = 1,
    pc_y: int = 2,
    labels: pd.Series | np.ndarray | None = None,
    title: str = "",
) -> plt.Figure:
    c1 = f"PC{int(pc_x)}"
    c2 = f"PC{int(pc_y)}"
    if c1 not in scores_df.columns or c2 not in scores_df.columns:
        raise ValueError(f"Scores DataFrame missing required columns {c1!r} and/or {c2!r}.")

    fig, ax = plt.subplots(figsize=(6, 5))
    x = scores_df[c1].to_numpy(dtype=float)
    y = scores_df[c2].to_numpy(dtype=float)

    if labels is None:
        ax.scatter(x, y, s=14, alpha=0.7, c="tab:blue", edgecolor="none")
    else:
        lab_series = pd.Series(labels, index=scores_df.index, copy=False)
        if lab_series.nunique(dropna=False) <= 20:
            for lab in pd.unique(lab_series):
                mask = lab_series == lab
                lab_text = "<NA>" if pd.isna(lab) else str(lab)
                ax.scatter(
                    x[mask.to_numpy()],
                    y[mask.to_numpy()],
                    s=16,
                    alpha=0.75,
                    label=lab_text,
                    edgecolor="none",
                )
            ax.legend(loc="best", fontsize=8, ncol=2)
        else:
            codes, uniques = pd.factorize(lab_series.astype(str))
            sc = ax.scatter(x, y, c=codes, cmap="viridis", s=14, alpha=0.75, edgecolor="none")
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Label code")
            cbar.ax.set_yticklabels([])
            del uniques

    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.axvline(0.0, color="0.75", lw=0.8)
    ax.set_xlabel(c1)
    ax.set_ylabel(c2)
    ax.set_title(title or f"Scores scatter: {c1} vs {c2}")
    fig.tight_layout()
    return fig


def _build_top_loadings_table(
    loadings_df: pd.DataFrame,
    *,
    pcs: list[int],
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pc in pcs:
        col = f"PC{int(pc)}"
        if col not in loadings_df.columns:
            continue
        vals = loadings_df[col]
        order = vals.abs().sort_values(ascending=False).head(int(top_n))
        for rank, (feature, abs_val) in enumerate(order.items(), start=1):
            loading = float(vals.loc[feature])
            rows.append(
                {
                    "pc": int(pc),
                    "feature": str(feature),
                    "loading": loading,
                    "abs_loading": float(abs_val),
                    "rank": int(rank),
                }
            )
    top_df = pd.DataFrame(rows)
    if top_df.empty:
        return top_df
    return top_df.sort_values(by=["pc", "rank"], kind="stable").reset_index(drop=True)


def plot_loadings_top(
    loadings_df: pd.DataFrame,
    *,
    pcs: list[int],
    top_n: int = 20,
    title: str = "Top absolute loadings",
) -> tuple[plt.Figure, pd.DataFrame]:
    top_df = _build_top_loadings_table(loadings_df, pcs=pcs, top_n=int(top_n))
    if top_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No loadings to plot", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        return fig, top_df

    n_panels = int(top_df["pc"].nunique())
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    for ax, pc in zip(axes[0], sorted(top_df["pc"].unique().tolist())):
        sub = top_df[top_df["pc"] == pc].sort_values("loading", ascending=True)
        colors = np.where(sub["loading"].to_numpy() >= 0.0, "tab:blue", "tab:red")
        ax.barh(sub["feature"].astype(str), sub["loading"].to_numpy(dtype=float), color=colors)
        ax.set_title(f"PC{int(pc)} top-{int(top_n)}")
        ax.set_xlabel("Loading value")
        ax.axvline(0.0, color="0.3", lw=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, top_df


def plot_biplot(
    scores_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    *,
    pc_x: int = 1,
    pc_y: int = 2,
    labels: pd.Series | np.ndarray | None = None,
    top_n: int = 15,
    arrow_scale: float | None = None,
    title: str = "Biplot",
) -> tuple[plt.Figure, dict[str, Any]]:
    c1 = f"PC{int(pc_x)}"
    c2 = f"PC{int(pc_y)}"
    if c1 not in scores_df.columns or c2 not in scores_df.columns:
        raise ValueError(f"Scores DataFrame missing required columns {c1!r} and/or {c2!r}.")
    if c1 not in loadings_df.columns or c2 not in loadings_df.columns:
        raise ValueError(f"Loadings DataFrame missing required columns {c1!r} and/or {c2!r}.")

    x = scores_df[c1].to_numpy(dtype=float)
    y = scores_df[c2].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    if labels is None:
        ax.scatter(x, y, s=14, alpha=0.5, c="0.55", edgecolor="none")
    else:
        lab_series = pd.Series(labels, index=scores_df.index, copy=False)
        if lab_series.nunique(dropna=False) <= 20:
            for lab in pd.unique(lab_series):
                mask = lab_series == lab
                lab_text = "<NA>" if pd.isna(lab) else str(lab)
                ax.scatter(
                    x[mask.to_numpy()],
                    y[mask.to_numpy()],
                    s=14,
                    alpha=0.65,
                    label=lab_text,
                    edgecolor="none",
                )
            ax.legend(loc="best", fontsize=8, ncol=2)
        else:
            codes, _ = pd.factorize(lab_series.astype(str))
            sc = ax.scatter(x, y, c=codes, cmap="viridis", s=14, alpha=0.6, edgecolor="none")
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Label code")

    plane = loadings_df[[c1, c2]].copy()
    norms = np.sqrt(np.square(plane[c1]) + np.square(plane[c2]))
    top = norms.sort_values(ascending=False).head(int(top_n))
    max_norm = float(top.max()) if len(top) else 1.0
    max_norm = max(max_norm, 1e-12)
    score_span = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1e-12)
    if arrow_scale is None:
        arrow_scale_used = 0.25 * score_span / max_norm
    else:
        arrow_scale_used = float(arrow_scale)

    for feature, _ in top.items():
        lx = float(plane.loc[feature, c1]) * arrow_scale_used
        ly = float(plane.loc[feature, c2]) * arrow_scale_used
        ax.arrow(0.0, 0.0, lx, ly, color="tab:red", alpha=0.8, width=0.0, head_width=0.03 * score_span)
        ax.text(lx, ly, str(feature), fontsize=7, color="tab:red")

    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axvline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel(c1)
    ax.set_ylabel(c2)
    ax.set_title(title)
    fig.tight_layout()
    return fig, {"arrow_scale": float(arrow_scale_used), "biplot_top_n": int(top_n)}


def build_interpretation_text(
    *,
    k_chosen: int,
    explained_variance_ratio_cum: np.ndarray,
    arrow_scale: float,
) -> str:
    cum = np.asarray(explained_variance_ratio_cum, dtype=float)
    k = int(k_chosen)
    k_var = float(cum[k - 1]) if cum.size >= k and k >= 1 else float("nan")
    text = (
        "PCA interpretation:\n"
        f"- Scree/cumulative EVR: k={k} reaches cumulative EVR={k_var:.4f}. "
        "For visualization, small k (often 2-3) is typical; for compression/denoising, choose k where cumulative EVR passes your target and the scree elbow stabilizes.\n"
        "- Loadings: larger absolute loading means stronger contribution of that feature to a PC. "
        "Features with same-sign loadings covary along that PC; opposite signs indicate trade-off.\n"
        "- Scores: each point/frame coordinate in PC space; distances in score space reflect similarity in projected representation.\n"
        f"- Biplot arrows: direction indicates feature increase in the PC plane, and relative length reflects representation strength in that plane (scaled by arrow_scale={arrow_scale:.4g} for readability)."
    )
    return text


def _save_figures_and_tables(
    *,
    figures: dict[str, plt.Figure],
    loadings_df: pd.DataFrame,
    top_loadings_df: pd.DataFrame,
    explained_variance_df: pd.DataFrame,
    save_dir: str | Path | None,
    save_prefix: str,
    save_plots: bool,
) -> dict[str, str]:
    if not bool(save_plots):
        return {}

    root = Path(save_dir) if save_dir is not None else (Path.cwd() / "pca_eda_outputs")
    fig_dir = root / "figures"
    table_dir = root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(save_prefix).strip("_")
    stem = prefix if prefix else "pca_eda"

    paths: dict[str, str] = {}
    for key, fig in figures.items():
        out = fig_dir / f"{stem}_{key}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        paths[f"fig_{key}"] = str(out)

    loadings_path = table_dir / f"{stem}_loadings.csv"
    top_loadings_path = table_dir / f"{stem}_top_loadings.csv"
    evr_path = table_dir / f"{stem}_explained_variance.csv"
    loadings_df.to_csv(loadings_path, index=True)
    top_loadings_df.to_csv(top_loadings_path, index=False)
    explained_variance_df.to_csv(evr_path, index=False)
    paths["table_loadings"] = str(loadings_path)
    paths["table_top_loadings"] = str(top_loadings_path)
    paths["table_explained_variance"] = str(evr_path)
    return paths


def run_pca_eda(
    data: pd.DataFrame | np.ndarray,
    features: list[str] | tuple[str, ...] | None = None,
    label_col: str | None = None,
    mode: str = "var_keep",
    var_keep: float = 0.95,
    n_components: int | None = None,
    scale: bool = True,
    missing: str = "median",
    top_n_loadings: int = 20,
    biplot_top_n: int = 15,
    score_pairs: tuple[tuple[int, int], ...] = ((1, 2), (1, 3)),
    save_dir: str | Path | None = None,
    save_prefix: str = "",
    save_plots: bool = True,
    random_state: int = 0,
) -> dict[str, Any]:
    del random_state  # Included for stable API; full SVD path is deterministic.

    prepared = _prepare_input_matrix(data, features=features, label_col=label_col)
    x_clean_df, y_clean, missing_meta = _apply_missing_policy(
        prepared.x_df, labels=prepared.labels, missing=missing
    )

    x_scaled, scaler, scale_meta = _apply_scaling(x_clean_df.to_numpy(dtype=np.float64), scale=bool(scale))
    fit = _fit_pca_model(
        x_scaled,
        mode=mode,
        var_keep=var_keep,
        n_components=n_components,
    )

    pca = fit["pca"]
    scores = np.asarray(fit["scores"], dtype=np.float64)
    evr = np.asarray(fit["evr"], dtype=np.float64)
    evr_cum = np.asarray(fit["evr_cum"], dtype=np.float64)
    k_chosen = int(fit["k_chosen"])
    available = int(fit["available_components"])

    score_cols = [f"PC{i+1}" for i in range(available)]
    scores_df = pd.DataFrame(scores, columns=score_cols, index=x_clean_df.index)

    loadings_mat = np.asarray(pca.components_.T, dtype=np.float64)
    loadings_df = pd.DataFrame(loadings_mat, index=x_clean_df.columns, columns=score_cols)

    explained_variance_df = pd.DataFrame(
        {
            "component": np.arange(1, available + 1, dtype=int),
            "explained_variance_ratio": evr,
            "explained_variance_ratio_cum": evr_cum,
        }
    )

    pcs_for_top = sorted(
        set(
            [1, 2, 3]
            + [int(pair[0]) for pair in score_pairs]
            + [int(pair[1]) for pair in score_pairs]
        )
    )
    pcs_for_top = [pc for pc in pcs_for_top if 1 <= pc <= available]

    title_prefix = str(save_prefix).replace("_", " ").strip()
    base_title = title_prefix if title_prefix else "PCA EDA"

    figures: dict[str, plt.Figure] = {}
    scree_fig = plot_scree_cumulative(evr, evr_cum, k_chosen=k_chosen, title=f"{base_title} | Scree")
    figures["scree"] = scree_fig

    for pair in score_pairs:
        px, py = int(pair[0]), int(pair[1])
        if px < 1 or py < 1 or px > available or py > available:
            continue
        key = f"scores_pc{px}_pc{py}"
        figures[key] = plot_scores_scatter(
            scores_df,
            pc_x=px,
            pc_y=py,
            labels=y_clean,
            title=f"{base_title} | Scores PC{px} vs PC{py}",
        )

    loadings_fig, top_loadings_df = plot_loadings_top(
        loadings_df,
        pcs=pcs_for_top,
        top_n=int(top_n_loadings),
        title=f"{base_title} | Top loadings",
    )
    figures["loadings_top"] = loadings_fig

    biplot_fig, biplot_meta = plot_biplot(
        scores_df,
        loadings_df,
        pc_x=1,
        pc_y=2,
        labels=y_clean,
        top_n=int(biplot_top_n),
        title=f"{base_title} | Biplot PC1 vs PC2",
    )
    figures["biplot_pc1_pc2"] = biplot_fig

    interpretation_text = build_interpretation_text(
        k_chosen=k_chosen,
        explained_variance_ratio_cum=evr_cum,
        arrow_scale=float(biplot_meta["arrow_scale"]),
    )
    print(interpretation_text)

    paths = _save_figures_and_tables(
        figures=figures,
        loadings_df=loadings_df,
        top_loadings_df=top_loadings_df,
        explained_variance_df=explained_variance_df,
        save_dir=save_dir,
        save_prefix=save_prefix,
        save_plots=bool(save_plots),
    )

    backend = str(plt.get_backend()).lower() if hasattr(plt, "get_backend") else ""
    show_inline = "agg" not in backend
    for fig in figures.values():
        fig.tight_layout()
        if show_inline:
            plt.show()

    meta: dict[str, Any] = {
        "source_type": prepared.source_type,
        "input_shape": prepared.input_shape,
        "n_samples": int(x_clean_df.shape[0]),
        "n_features": int(x_clean_df.shape[1]),
        "k_chosen": int(k_chosen),
        "explained_variance_ratio_sum_k": float(evr_cum[k_chosen - 1]),
        "available_components": int(available),
        "arrow_scale": float(biplot_meta["arrow_scale"]),
        "used_label_col": str(label_col) if (label_col and prepared.labels is not None) else "",
        "missing": missing_meta,
        "scaling": scale_meta,
        "save_prefix": str(save_prefix),
    }

    return {
        "pca": pca,
        "scores_df": scores_df,
        "loadings_df": loadings_df,
        "top_loadings_df": top_loadings_df,
        "explained_variance_df": explained_variance_df,
        "figures": figures,
        "interpretation_text": interpretation_text,
        "meta": meta,
        "paths": paths,
        "scaler": scaler,
    }
