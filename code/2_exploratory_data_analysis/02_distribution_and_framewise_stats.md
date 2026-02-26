# 02_distribution_and_framewise_stats.ipynb

## Key variables (produced)
- `session_id`--> read from each .npz file name
- `abs_across_mean`--> 
- `abs_across_std`

## Functions
- `_eda_subject_root`
- `_flatten_for_hist`
- `_mode_transform`
- `_subject_raw_range`
- `compute_framewise_stats`

## Plots
- Per-acquisition value distribution panel (1x3 histograms) comparing modes: `raw`, `zscore`, `mean_divide`
- Per-acquisition framewise statistics triplet (3 stacked line plots over frame index) for metrics such as central tendency, spread, and `dvars`
- Per-acquisition frame-to-frame difference map panel (2x2): mean frame context, mean diff, diff std, and mean absolute diff
- Across-acquisition frame-to-frame difference map panel (2x2) aggregated by mode
