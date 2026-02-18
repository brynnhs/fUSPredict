# fUSPredict

Forecasting baseline functional ultrasound (fUS) dynamics with classical autoregressive baselines.

This repository currently uses a notebook-first pipeline with four main stages:

1. `code/1_processing.ipynb`
2. `code/2_exploratory_data_analysis.ipynb`
3. `code/3_dataset_creation.ipynb`
4. `code/4_autoregression_modeling.ipynb`

## Environment

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Main dependencies include `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, `scikit-learn`, `opencv-python`, and `statsmodels`.

## Repository layout

- `sourcedata/`: input data (session `.mat` files and labels)
- `derivatives/preprocessing/`: outputs from baseline extraction and normalization
- `derivatives/modeling/phase1_baselines/`: outputs from autoregressive baselines
- `code/utils/`: reusable helpers and dataset/modeling utilities
- `code/1_processing.ipynb`: preprocessing and QC
- `code/2_exploratory_data_analysis.ipynb`: exploratory visual/statistical checks
- `code/3_dataset_creation.ipynb`: split manifest creation
- `code/4_autoregression_modeling.ipynb`: phase-1 baseline modeling

## Main pipeline (detailed)

### Stage 1: Baseline extraction and preprocessing (`1_processing.ipynb`)

Purpose: convert raw session files into per-session baseline frame caches, normalized variants, and QC outputs.

Input assumptions:

- Subject folders exist under `sourcedata/`.
- Session data files follow `Datas_*.mat` naming.
- Label files follow `Label_pauses_*.mat` naming.
- Baseline frames are identified by label value `-1`.

Core operations:

- Discover subject sessions.
- Read fUS frames from `.mat` (`Data` or `Datas` key, then `fus/frame` fields).
- Read labels from matching label file.
- Align frame/label length with `mismatch(...)` truncation.
- Select baseline-only frames via `labels == -1`.
- Save one compressed cache per session to:
  - `derivatives/preprocessing/<subject>/baseline_only/baseline_<session>.npz`

Saved baseline cache fields include:

- `frames`: baseline frames, shape `(T_baseline, H, W)`, float32
- `original_indices`: frame indices in the original acquisition
- `session_id`
- `source_fps` when available
- source file metadata and frame counts

Normalization step:

- Build `baseline_only_normalized/<mode>/` for:
  - `mean_divide`: `(x - mean_t) / mean_t` per pixel over time
  - `zscore`: `(x - mean_t) / std_t` per pixel over time
- Optional ROI-aware normalization path exists.
- Outputs saved as:
  - `derivatives/preprocessing/<subject>/baseline_only_normalized/mean_divide/baseline_<session>_mean_divide.npz`
  - `derivatives/preprocessing/<subject>/baseline_only_normalized/zscore/baseline_<session>_zscore.npz`

Visualization and QC in this notebook:

- Subject-level triplet video (raw, mean_divide, zscore) for a representative baseline session
  (default: most stable session by median frame-to-frame absolute difference):
  - `derivatives/preprocessing/<subject>/<subject>_baseline_triplet_raw_mean_zscore.mp4`
- ACF signal sanity summary across saved modes:
  - `derivatives/preprocessing/<subject>/acf_qc_summary.csv`

### Stage 2: Exploratory analysis (`2_exploratory_data_analysis.ipynb`)

Purpose: inspect baseline behavior before modeling.

Main checks include:

- Pixel and patch-level timecourses
- Sample frame and patch visualizations
- Distribution summaries over time (mean/median/max)
- Pixel stability and coefficient-of-variation style maps
- Frame-to-frame difference checks
- Spatiotemporal dependence and PSD/autocorrelation diagnostics
- Spatial autocorrelation maps per session and across sessions (mean/std aggregates and per-session montage), with neighborhood settings (`4` and `8`)

This stage is diagnostic and helps validate preprocessing choices (for example whether z-score normalization stabilizes dynamics and suppresses intensity-scale confounds).

### Stage 3: Dataset split manifests (`3_dataset_creation.ipynb`)

Purpose: create deterministic acquisition-level train/test manifests consumed by `FUSForecastWindowDataset`.

Manifest utility functions support:

- Single-cache split (`create_split_manifest`)
- Multi-subject combined split with subject metadata (`create_split_manifest_multi`)
- Per-subject manifests (`create_split_manifests_per_subject`)
- Two-subject held-out train/test direction swaps (`create_subject_heldout_manifests`)

Key behavior:

- Splits are acquisition-wise random with fixed seed.
- Split ratio defaults to `0.8`.
- Paths are sorted for deterministic output ordering.

Current notebook usage highlights:

- `DATA_MODE = "zscore"` by default.
- Combined manifest saved to:
  - `derivatives/preprocessing/splits_multi.json`
- Single-subject secundo manifest saved to:
  - `derivatives/preprocessing/splits_single_secundo.json`

The single-subject manifest is what `4_autoregression_modeling.ipynb` uses by default.

### Stage 4: Autoregressive baseline modeling (`4_autoregression_modeling.ipynb`)

Purpose: train/evaluate phase-1 classical forecasting baselines on sliding windows.

Dataset construction:

- Uses `FUSForecastWindowDataset` with:
  - `window_size W = 8`
  - `pred_horizon K = 1` for one-step training/eval
  - `stride S = 1`
- Default manifest:
  - `derivatives/preprocessing/splits_single_secundo.json`

Important: model space follows manifest data

- If manifest paths point to zscore caches, training and prediction are in zscore frame space.
- No extra normalization is applied by `FUSForecastWindowDataset` beyond shape handling/casting.
- The notebook now prints explicit data-space sanity checks:
  - inferred manifest mode (`raw`/`mean_divide`/`zscore`) from cache paths
  - float-space GT/Pred stats (`mean`, `std`) on sample test windows
  - zscore cache temporal per-pixel sanity summaries when in zscore mode
  - near-zero-variance pixel fractions on sample training acquisitions

Models trained:

- Persistence baseline: next frame equals last context frame
- Pixel AR(p) with ridge, `p in [1, 2, 5]`
- PCA + VAR(p), `d in [256, 512, 1024]`, `p in [1, 2, 5]`
- PCA + diagonal AR(p), same `d` and `p` grid

Core outputs:

- Model parameter artifacts (`.npz` for each configuration)
- Aggregate metric tables:
  - `derivatives/modeling/phase1_baselines/metrics.csv`
  - `derivatives/modeling/phase1_baselines/metrics_standardized.csv` (optional; see below)
- Summary tables/plots for leaderboard and horizon behavior
- Residual latent autocorrelation diagnostics
- Best/worst model triplet videos:
  - `derivatives/modeling/phase1_baselines/videos/`

Metric-space and residual conventions:

- `metrics.csv` is computed directly in the native data space loaded by the manifest
  (raw if manifest points to raw caches, zscore if manifest points to zscore caches).
- `metrics_standardized.csv` (when enabled) is evaluation-time per-window standardization:
  each target frame is z-scored by its own mean/std, and prediction is transformed with the same target stats before metric computation.
- If the manifest already points to normalized caches (`zscore` or `mean_divide`), the notebook disables this standardized pass by default to avoid confusing interpretation.
- Metric computation is done in float space; display-space (`uint8`) panels are not used for RMSE/MAE/R2.
- Residuals for quantitative diagnostics are float residuals:
  `residual = pred - gt` (signed, unit-preserving).
- Residual panels in exported videos are display mappings only:
  signed float residuals are robust-scaled to `uint8` for visualization and should not be interpreted as the numeric error itself.

Horizon rollout diagnostics:

- Evaluates selected models on horizons `[1, 2, 5, 10, 20]` using iterative rollout.
- Exports horizon RMSE curves and delta-vs-baseline curves.

Non-PCA statistical comparison block:

- Compares persistence and pixel AR variants at acquisition level.
- Computes paired Wilcoxon and paired t-tests.
- Saves corrected p-values and effect summaries to:
  - `derivatives/modeling/phase1_baselines/non_pca_model_stats_acq_level.csv`
  - `derivatives/modeling/phase1_baselines/stats_plots_non_pca/`

## Data products at a glance

Preprocessing outputs:

- `derivatives/preprocessing/<subject>/baseline_only/*.npz`
- `derivatives/preprocessing/<subject>/baseline_only_normalized/{mean_divide,zscore}/*.npz`
- `derivatives/preprocessing/<subject>/acf_qc_summary.csv`
- `derivatives/preprocessing/<subject>/<subject>_baseline_triplet_raw_mean_zscore.mp4`
- `derivatives/preprocessing/splits_*.json`

Modeling outputs:

- `derivatives/modeling/phase1_baselines/*.npz` model params
- `derivatives/modeling/phase1_baselines/metrics*.csv`
- `derivatives/modeling/phase1_baselines/phase1_metrics_table*.csv/.png`
- `derivatives/modeling/phase1_baselines/error_vs_horizon*.png/.pdf`
- `derivatives/modeling/phase1_baselines/delta_rmse_vs_pixel_ar_p5*.png/.pdf`
- `derivatives/modeling/phase1_baselines/percent_worse_vs_pixel_ar_p5*.png/.pdf`
- `derivatives/modeling/phase1_baselines/residual_acf_best_model*.png/.pdf`
- `derivatives/modeling/phase1_baselines/videos/*.mp4`
- `derivatives/modeling/phase1_baselines/stats_plots_non_pca/*`

## Recommended execution order

1. Run `code/1_processing.ipynb` to generate baseline caches and normalized variants.
2. Run `code/2_exploratory_data_analysis.ipynb` to validate signal behavior.
3. Run `code/3_dataset_creation.ipynb` to generate manifests for your chosen data mode.
4. Run `code/4_autoregression_modeling.ipynb` to train/evaluate phase-1 baselines and export diagnostics.

## Notes

- Splits are acquisition-level by design; this reduces leakage risk compared with frame-level random splits.
- Normalization in stage 1 is per session over time, per pixel.
- In stage 4, ensure your chosen manifest and your interpretation of metric spaces are aligned (raw vs zscore data source).
- PCA baselines are fit on train acquisitions/windows only; test data are projected/evaluated using the train-fitted basis.
