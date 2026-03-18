from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CUSTOM_PALETTE = [
    "#D0E9F1",
    "#A3D0D4",
    "#48A7C8",
    "#041C3C",
    "#2A356B",
    "#565AA0",
]


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _robust_limits(x: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> tuple[float, float]:
    arr = np.asarray(x, dtype=np.float32)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, float(q_low)))
    hi = float(np.percentile(vals, float(q_high)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def _robust_abs_limit(x: np.ndarray, q: float = 99.5, eps: float = 1e-8) -> float:
    arr = np.abs(np.asarray(x, dtype=np.float32))
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 1.0
    limit = float(np.percentile(vals, float(q)))
    if limit <= float(eps) or not np.isfinite(limit):
        limit = float(np.max(vals))
    if limit <= float(eps) or not np.isfinite(limit):
        return 1.0
    return limit


def _finalize_plot(fig: plt.Figure, save_paths: list[str | Path] | None, show_inline: bool) -> None:
    fig.tight_layout()
    if save_paths:
        for path in save_paths:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
    if show_inline:
        plt.show()
    else:
        plt.close(fig)


def _pick_evenly_spaced_indices(n_items: int, max_items: int) -> list[int]:
    if n_items <= 0:
        return []
    count = max(1, min(int(max_items), int(n_items)))
    if count == 1:
        return [0]
    return np.unique(np.linspace(0, n_items - 1, num=count, dtype=np.int64)).astype(int).tolist()


def rollout_predict(
    predict_step_fn,
    context: Any,
    n_steps: int,
    *,
    target: Any | None = None,
    mode: str = "multistep",
) -> np.ndarray:
    ctx = _to_numpy(context)
    tgt = None if target is None else _to_numpy(target)
    preds = None
    for step_i in range(int(n_steps)):
        step = np.asarray(predict_step_fn(ctx))
        if preds is None:
            preds = np.empty((int(n_steps),) + step.shape[1:], dtype=step.dtype)
        preds[step_i] = step[0]
        if mode == "multistep":
            ctx = np.concatenate([ctx[1:], step], axis=0)
        elif mode == "onestep":
            if tgt is None:
                raise ValueError("onestep rollout requires target sequence.")
            gt_step = tgt[step_i : step_i + 1]
            ctx = np.concatenate([ctx[1:], gt_step], axis=0)
        else:
            raise ValueError(f"Unknown rollout mode: {mode}")
    if preds is None:
        raise RuntimeError("No predictions were produced.")
    return preds


def plot_horizon_metric_grid(
    horizon_df: pd.DataFrame,
    *,
    title_prefix: str,
    save_paths: list[str | Path] | None = None,
    show_inline: bool = False,
    exclude_models: list[str] | None = None,
    metrics: list[tuple[str, str]] | None = None,
) -> plt.Figure:
    if horizon_df.empty:
        raise ValueError("horizon_df is empty.")
    required = {"model", "horizon"}
    missing = required.difference(horizon_df.columns)
    if missing:
        raise ValueError(f"horizon_df missing required columns: {sorted(missing)}")

    metric_specs = metrics or [
        ("MSE_mean", "MSE"),
        ("RMSE_mean", "RMSE"),
        ("MAE_mean", "MAE"),
        ("R2_mean", "R2"),
    ]
    available = [(col, label) for col, label in metric_specs if col in horizon_df.columns]
    if not available:
        raise ValueError("No requested metric columns are present in horizon_df.")

    exclude = {str(x) for x in (exclude_models or [])}
    grouped = {
        str(model_name): grp.sort_values("horizon")
        for model_name, grp in horizon_df.groupby("model", sort=False)
        if str(model_name) not in exclude
    }
    if not grouped:
        raise ValueError("No models remain after filtering.")

    n_panels = len(available)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.2 * nrows), squeeze=False)

    for ax, (metric_col, metric_label) in zip(axes.ravel(), available):
        for model_name, grp in grouped.items():
            ax.plot(grp["horizon"].values, grp[metric_col].values, marker="o", linewidth=1.8, label=model_name)
        ax.set_title(f"{metric_label} vs horizon")
        ax.set_xlabel("Horizon")
        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)

    for ax in axes.ravel()[len(available) :]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=9, frameon=False)
        fig.suptitle(title_prefix, y=1.02)
    else:
        fig.suptitle(title_prefix)

    _finalize_plot(fig, save_paths=save_paths, show_inline=show_inline)
    return fig


def plot_combo_best_rmse(
    combo_summary_df: pd.DataFrame,
    *,
    save_paths: list[str | Path] | None = None,
    show_inline: bool = False,
) -> plt.Figure:
    if combo_summary_df.empty:
        raise ValueError("combo_summary_df is empty.")
    required = {"combo_id", "best_rmse", "best_model"}
    missing = required.difference(combo_summary_df.columns)
    if missing:
        raise ValueError(f"combo_summary_df missing required columns: {sorted(missing)}")

    plot_df = combo_summary_df.copy().sort_values(["best_rmse", "combo_id"]).reset_index(drop=True)
    model_names = plot_df["best_model"].astype(str).tolist()
    unique_models = list(dict.fromkeys(model_names))
    color_map = {name: CUSTOM_PALETTE[i % len(CUSTOM_PALETTE)] for i, name in enumerate(unique_models)}
    colors = [color_map[name] for name in model_names]

    fig, ax = plt.subplots(figsize=(max(8.5, 1.6 * len(plot_df) + 2.0), 4.8))
    ax.bar(plot_df["combo_id"], plot_df["best_rmse"], color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("Best RMSE by preprocessing combo")
    ax.set_xlabel("Preprocessing combo")
    ax.set_ylabel("Best RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=name, markerfacecolor=color_map[name], markersize=10)
        for name in unique_models
    ]
    if handles:
        ax.legend(
            handles=handles,
            title="Best model",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=False,
            fontsize=9,
            title_fontsize=10,
        )
        fig.subplots_adjust(right=0.76)

    _finalize_plot(fig, save_paths=save_paths, show_inline=show_inline)
    return fig


def plot_rollout_comparison_examples(
    dataset,
    model_registry: dict[str, Any],
    horizons: list[int] | tuple[int, ...],
    output_dir: str | Path,
    *,
    rollout_mode: str = "multistep",
    example_indices: list[int] | None = None,
    max_examples: int = 3,
    max_models: int | None = 4,
    show_inline: bool = False,
) -> list[Path]:
    if len(model_registry) == 0:
        raise ValueError("model_registry must not be empty.")

    horizon_list = sorted({int(h) for h in horizons if int(h) > 0})
    if len(horizon_list) == 0:
        raise ValueError("At least one positive horizon is required.")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if example_indices is None:
        example_indices = _pick_evenly_spaced_indices(len(dataset), max_examples)
    if len(example_indices) == 0:
        raise ValueError("No example indices available for plotting.")

    model_names = list(model_registry.keys())
    if max_models is not None:
        model_names = model_names[: int(max_models)]

    saved_paths: list[Path] = []
    max_h = max(horizon_list)

    for example_idx in example_indices:
        item = dataset[int(example_idx)]
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError("Dataset item must be (context, target) or (context, target, meta).")

        context = _to_numpy(item[0])
        target = _to_numpy(item[1])
        if target.shape[0] < max_h:
            raise ValueError(
                f"Target horizon {target.shape[0]} is shorter than requested max horizon {max_h}."
            )

        gt_frames = np.asarray(target[[h - 1 for h in horizon_list], 0], dtype=np.float32)
        preds_by_model: dict[str, np.ndarray] = {}
        residuals: list[np.ndarray] = []

        for model_name in model_names:
            pred = rollout_predict(
                model_registry[model_name],
                context,
                max_h,
                target=target,
                mode=rollout_mode,
            )
            pred_sel = np.asarray(pred[[h - 1 for h in horizon_list], 0], dtype=np.float32)
            preds_by_model[model_name] = pred_sel
            residuals.append(pred_sel - gt_frames)

        vmin, vmax = _robust_limits(np.concatenate([gt_frames.reshape(-1), *[p.reshape(-1) for p in preds_by_model.values()]]))
        residual_scale = _robust_abs_limit(np.stack(residuals, axis=0))

        # Prediction comparison grid.
        nrows = 1 + len(model_names)
        ncols = len(horizon_list)
        fig_pred, axes_pred = plt.subplots(
            nrows,
            ncols,
            figsize=(2.4 * ncols, 2.4 * nrows),
            squeeze=False,
        )
        for col_idx, horizon in enumerate(horizon_list):
            ax = axes_pred[0, col_idx]
            ax.imshow(gt_frames[col_idx], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"GT h={horizon}")
            ax.axis("off")

        for row_idx, model_name in enumerate(model_names, start=1):
            pred_sel = preds_by_model[model_name]
            for col_idx, horizon in enumerate(horizon_list):
                pred_frame = pred_sel[col_idx]
                rmse = float(np.sqrt(np.mean((pred_frame - gt_frames[col_idx]) ** 2)))
                ax = axes_pred[row_idx, col_idx]
                ax.imshow(pred_frame, cmap="gray", vmin=vmin, vmax=vmax)
                if col_idx == 0:
                    ax.set_ylabel(model_name, rotation=0, ha="right", va="center", labelpad=30)
                ax.set_xlabel(f"RMSE={rmse:.3f}", fontsize=8)
                ax.axis("off")

        fig_pred.suptitle(f"Rollout comparison | window={example_idx} | mode={rollout_mode}")
        pred_path = out_root / f"rollout_compare_window_{int(example_idx):04d}.png"
        _finalize_plot(fig_pred, save_paths=[pred_path], show_inline=show_inline)
        saved_paths.append(pred_path)

        # Residual comparison grid.
        fig_res, axes_res = plt.subplots(
            len(model_names),
            ncols,
            figsize=(2.4 * ncols, 2.4 * max(1, len(model_names))),
            squeeze=False,
        )
        for row_idx, model_name in enumerate(model_names):
            pred_sel = preds_by_model[model_name]
            for col_idx, horizon in enumerate(horizon_list):
                resid = pred_sel[col_idx] - gt_frames[col_idx]
                rmse = float(np.sqrt(np.mean(resid ** 2)))
                ax = axes_res[row_idx, col_idx]
                ax.imshow(resid, cmap="bwr", vmin=-residual_scale, vmax=residual_scale)
                if row_idx == 0:
                    ax.set_title(f"h={horizon}")
                if col_idx == 0:
                    ax.set_ylabel(model_name, rotation=0, ha="right", va="center", labelpad=30)
                ax.set_xlabel(f"RMSE={rmse:.3f}", fontsize=8)
                ax.axis("off")

        fig_res.suptitle(f"Residuals (pred - gt) | window={example_idx} | mode={rollout_mode}")
        resid_path = out_root / f"rollout_residuals_window_{int(example_idx):04d}.png"
        _finalize_plot(fig_res, save_paths=[resid_path], show_inline=show_inline)
        saved_paths.append(resid_path)

    return saved_paths
