# 02_distribution_and_framewise_stats.ipynb

## Key variables
| Variable | Type | Meaning |
| session_id |string  |read from each .npz file name |
| fdiff_robust_pctl | tuple[float, float] | Low/high percentiles for robust color limits |
| spatial_eps | float | Small epsilon to avoid divide-by-zero edge cases |

- `session_id`| string | read from each .npz file name
- `abs_across_mean`| | get the absolute mean from np.nanmean, this is to drop direction and keep the magnitude, as the sign can cancel out across acquistions. for single acquistion plots, the signed diff is used
- `abs_across_std`| | get the absolute std from np.nanmeanthis, is to drop direction and keep the magnitude, as the sign can cancel out across acquistions
- `fdiff_robust_pctl` | | percentile range for plot limits for the frame to frame diff maps
- `spatial_eps` | |
- `spatial_min_valid_samples` | |
- `DIST_RNG_SEED` | |
- `DIST_RAW_RANGE_PERCENTILES` | |
- `DIST_HIST_FIGSIZE` | |
- `DIST_SAVE_DPI` | |
- `TIME_STATS_FIGSIZE` | |
- `TIME_STATS_SAVE_DPI` | |
- `FRAME_DIFF_METHOD` | |
- `FDIFF_ROOT_SUBDIR` | |
- `FRAME_DIFF_FIGSIZE` | |
- `FRAME_DIFF_SAVE_DPI` | |
- `GLOBAL_SIGNAL_MODES` | |

    
# Distribution config (shared)
dist_cfg = eda_cfg["distribution"]

DIST_N_BINS = int(dist_cfg["n_bins"])
DIST_MAX_POINTS = int(dist_cfg["max_points"])
DIST_MODE_ORDER = list(analysis_modes)
DIST_MODE_RANGES = {k: tuple(v) for k, v in dist_cfg["mode_ranges"].items()}

## Functions
- `_eda_subject_root` --> set the root for EDA outputs (histograms, framewise stats, etc.)
- `_flatten_for_hist`--> flatten the array and subsample if there are too many points, for histogram plotting
- `_mode_transform`--> Mode-specific transformation to match saved session frames TODO: maybe add this to utils
- `_subject_raw_range`--> compute subject-specific robust range for raw mode based on baseline sessions uses DIST_RAW_RANGE_PERCENTILES set at the beginning of the cell
- `compute_framewise_stats`--> 

## Plots
- Per-acquisition value distribution panel (1x3 histograms) comparing modes: `raw`, `zscore`, `mean_divide`
- Per-acquisition framewise statistics triplet (3 stacked line plots over frame index) for metrics such as central tendency, spread, and `dvars`
- Per-acquisition frame-to-frame difference map panel (2x2): mean frame context, mean diff, diff std, and mean absolute diff
- Across-acquisition frame-to-frame difference map panel (2x2) aggregated by mode
