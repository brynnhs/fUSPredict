"""
Author: Leo Sperber, 2025
"""
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F
import os
import glob
from scipy import signal
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MatplotlibPath
from pathlib import Path
"""
Author: Brynn Harris-Shanks, 2026
"""
# Function to extract and save the baseline frames --> for fUS Predict baseline
def extract_baseline_frames_from_mat(mat_dict):
    """
    Extract frames from .mat file structure.
    Handles both 'Data' and 'Datas' keys.
    Returns frames in [T, H, W] format.
    """
    key = "Data" if "Data" in mat_dict else ("Datas" if "Datas" in mat_dict else None)
    if key is None:
        raise KeyError("Neither 'Data' nor 'Datas' found in .mat file")

    try:
        fus_struct = mat_dict[key]["fus"][0, 0]
        frames = fus_struct["frame"][0, 0]
    except Exception as e:
        raise KeyError("Failed to access mat[key]['fus'][0,0]['frame'][0,0]") from e

    frames = np.asarray(frames)
    
    # Normalize to [T, H, W]
    # Heuristic: if last dim is the largest, assume [H, W, T]
    if frames.ndim == 3:
        if frames.shape[2] > frames.shape[0] and frames.shape[2] > frames.shape[1]:
            frames = np.transpose(frames, (2, 0, 1))
    
    return frames

def load_label_file(label_path):
    """Load labels from Label_pauses_*.mat file"""
    lab = scipy.io.loadmat(label_path)
    labels = lab['Datas']['Label'][0, 0]
    labels_arr = np.asarray(labels).squeeze()
    return labels_arr

def extract_and_save_baseline(fus_path, label_path, output_dir):
    """
    Extract baseline frames from a session and save as .npz
    Uses mismatch() to align frames and labels.
    
    Args:
        fus_path: Path to Datas_Se*.mat file
        label_path: Path to Label_pauses_Se*.mat file
        output_dir: Directory to save baseline .npz files
    
    Returns:
        Path to saved baseline file, or None if error
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract date code for naming
    stem = Path(fus_path).stem  # e.g., "Datas_Se01072020"
    date_code = stem.replace("Datas_", "")
    baseline_output_path = os.path.join(output_dir, f"baseline_{date_code}.npz")
    
    # Skip if already exists
    if os.path.exists(baseline_output_path):
        print(f"⏭️  Skipping {date_code} (already exists)")
        return baseline_output_path
    
    try:
        # Load frames
        mat = scipy.io.loadmat(fus_path)
        frames = extract_baseline_frames_from_mat(mat)
        
        # Load labels
        labels_arr = load_label_file(label_path)
        
        # Use helper function to align frames and labels
        frames, labels_arr = mismatch(frames, labels_arr)
        
        # Extract baseline frames
        baseline_mask = (labels_arr == -1)
        baseline_frames = frames[baseline_mask]  # Shape: (T_baseline, H, W)
        baseline_indices = np.where(baseline_mask)[0]  # Original frame indices
        
        if len(baseline_frames) == 0:
            print(f"⚠️  Session {date_code}: No baseline frames found (label == -1)")
            return None
        
        print(f"✅ Session {date_code}: {len(baseline_frames)}/{len(frames)} frames are baseline "
              f"({len(baseline_frames)/len(frames)*100:.1f}%)")
        
        # Save baseline data with metadata
        np.savez_compressed(
            baseline_output_path,
            frames=baseline_frames.astype(np.float32),  # Save as float32 to save space
            original_indices=baseline_indices,  # Which frames in original file were baseline
            session_id=date_code,
            source_fus_file=os.path.basename(fus_path),
            source_label_file=os.path.basename(label_path),
            n_total_frames=len(frames),
            n_baseline_frames=len(baseline_frames),
            spatial_shape=baseline_frames.shape[1:],  # (H, W)
            dtype=str(baseline_frames.dtype)
        )
        
        return baseline_output_path
        
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(fus_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_all_baseline_files(data_directory, output_dir):
    """
    Extract baseline from all sessions.
    
    Args:
        data_directory: Directory containing Datas_*.mat and Label_pauses_*.mat files
        output_dir: Directory to save baseline .npz files
    
    Returns:
        List of paths to saved baseline files
    """
    # Find all fUS files
    fus_files = sorted(glob.glob(os.path.join(data_directory, "Datas_*.mat")))
    
    if len(fus_files) == 0:
        print(f"⚠️  No Datas_*.mat files found in {data_directory}")
        return []
    
    print(f"Found {len(fus_files)} fUS files to process\n")
    
    baseline_files = []
    for fus_path in fus_files:
        # Find matching label file
        stem = Path(fus_path).stem
        date_code = stem.replace("Datas_", "")
        label_path = os.path.join(data_directory, f"Label_pauses_{date_code}.mat")
        
        if not os.path.exists(label_path):
            print(f"⚠️  No label file for {os.path.basename(fus_path)}")
            continue
        
        baseline_path = extract_and_save_baseline(fus_path, label_path, output_dir)
        if baseline_path:
            baseline_files.append(baseline_path)
    
    print(f"\n Extracted baseline from {len(baseline_files)}/{len(fus_files)} sessions")
    return baseline_files

def load_baseline_session(baseline_path):
    """
    Load a single session's baseline data.
    
    Returns:
        dict with 'frames' (T, H, W) and metadata
    """
    data = np.load(baseline_path, allow_pickle=True)
    return {
        'frames': data['frames'],  # (T, H, W)
        'session_id': str(data['session_id']),
        'original_indices': data['original_indices'],
        'metadata': {k: data[k] for k in data.files if k not in ['frames', 'original_indices', 'session_id']}
    }

def load_all_baseline(baseline_dir):
    """
    Load all baseline sessions.
    
    Returns:
        List of dicts, each containing a session's baseline data
    """
    baseline_files = sorted(glob.glob(os.path.join(baseline_dir, "baseline_*.npz")))
    
    if len(baseline_files) == 0:
        print(f"⚠️  No baseline_*.npz files found in {baseline_dir}")
        return []
    
    all_sessions = []
    for path in baseline_files:
        try:
            session_data = load_baseline_session(path)
            all_sessions.append(session_data)
        except Exception as e:
            print(f"❌ Error loading {os.path.basename(path)}: {e}")
    
    print(f"✅ Loaded {len(all_sessions)} baseline sessions")
    return all_sessions
# Function to plot raw fUS intensity over time with label shading
def plot_fus_timecourse_with_labels(
    fus_path_or_dir,
    label_path_or_dir=None,
    label_colors=None,
    alpha=0.15,
    sessions=None
):
    """
    Plot raw fUS timecourse(s) with behavioral label shading.

    Accepted usage:
    1) Single session file paths:
       plot_fus_timecourse_with_labels(".../Datas_SeXXXX.mat", ".../Label_pauses_SeXXXX.mat")
    2) Whole folder (labels in same folder):
       plot_fus_timecourse_with_labels(data_directory)
       plot_fus_timecourse_with_labels(data_directory, data_directory)

    Session selection (directory mode):
       - sessions=None or sessions="all": plot all matched sessions
       - sessions="Se01072020": plot one session
       - sessions=["Se01072020", "Se02072020"]: plot selected sessions
    """
    if label_colors is None:
        label_colors = {-1: 'lightblue', 0: 'red', 1: 'lightgreen'}

    # Case 1: folder input -> find and match sessions
    if os.path.isdir(fus_path_or_dir):
        fus_dir = fus_path_or_dir
        label_dir = label_path_or_dir if label_path_or_dir is not None else fus_dir

        fus_files = sorted(glob.glob(os.path.join(fus_dir, "Datas_*.mat")))
        if len(fus_files) == 0:
            raise FileNotFoundError(f"No Datas_*.mat files found in: {fus_dir}")

        matched_sessions = []
        for fus_path in fus_files:
            session_id = Path(fus_path).stem.replace("Datas_", "")
            label_path = os.path.join(label_dir, f"Label_pauses_{session_id}.mat")

            if not os.path.exists(label_path):
                fallback = sorted(glob.glob(os.path.join(label_dir, f"Label*{session_id}.mat")))
                if len(fallback) > 0:
                    label_path = fallback[0]

            if os.path.exists(label_path):
                matched_sessions.append((fus_path, label_path))
            else:
                print(f"Warning: missing label file for {os.path.basename(fus_path)}")

        if len(matched_sessions) == 0:
            raise FileNotFoundError(f"No matching label files found in: {label_dir}")

        # Select which sessions to plot
        if sessions is None or sessions == "all":
            selected_sessions = matched_sessions
        else:
            if isinstance(sessions, str):
                requested_ids = {sessions}
            elif isinstance(sessions, (list, tuple, set, np.ndarray)):
                requested_ids = {str(s) for s in sessions}
            else:
                raise ValueError(
                    "sessions must be None, 'all', a session string, or a list/tuple/set of session strings."
                )

            selected_sessions = [
                (fus_path, label_path)
                for fus_path, label_path in matched_sessions
                if Path(fus_path).stem.replace("Datas_", "") in requested_ids
            ]

            found_ids = {
                Path(fus_path).stem.replace("Datas_", "")
                for fus_path, _ in selected_sessions
            }
            missing_ids = requested_ids - found_ids
            for missing in sorted(missing_ids):
                print(f"Warning: requested session not found: {missing}")

            if len(selected_sessions) == 0:
                raise FileNotFoundError("None of the requested sessions were found.")

    # Case 2: direct file input
    else:
        if label_path_or_dir is None:
            raise ValueError(
                "When fus_path_or_dir is a file, label_path_or_dir must be the label file path."
            )
        selected_sessions = [(fus_path_or_dir, label_path_or_dir)]

    # Plot each selected session
    for fus_path, label_path in selected_sessions:
        mat = scipy.io.loadmat(fus_path)
        frames = extract_baseline_frames_from_mat(mat)
        labels_arr = load_label_file(label_path)
        frames, labels_arr = mismatch(frames, labels_arr)

        fus_ts = frames.reshape(frames.shape[0], -1).mean(axis=1)
        session_name = Path(fus_path).stem.replace("Datas_", "")

        fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
        add_label_shading(ax, labels_arr, label_colors, alpha=alpha)
        ax.plot(fus_ts, color='tab:blue', lw=1.0, label='Raw fUS mean intensity')
        ax.set_title(f"Raw fUS Timecourse - {session_name}")
        ax.set_xlabel('Frame')
        ax.set_ylabel('Raw fUS intensity (a.u.)')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        plt.show()

"""
Author: Leo Sperber, 2025
"""
# Function to handle mismatch between images and labels --> counts # of frames and labels and makes them the same length
def mismatch(images, labels_arr):
    # Handle mismatch: Shave to the minimum length instead of discarding
    if images.shape[0] != len(labels_arr):
        min_len = min(images.shape[0], len(labels_arr))
        original_images = images.shape[0]
        original_labels = len(labels_arr)
        images = images[:min_len]
        labels_arr = labels_arr[:min_len]
        print(f"  - MISMATCH: Images={original_images}, Labels={original_labels}. Shaving to {min_len} frames.")
    else:
        print(f"  - Match confirmed: {images.shape[0]} frames and labels.")
    return images, labels_arr

# Function to compute %CBV on original size (H_var, 128) --> we do this by subtracting the baseline from the images and then dividing by the baseline
def delta_cbv(images, labels_arr, use_log=True, robust=True):
        # Compute %CBV on original size (H_var, 128)
    baseline_mask = (labels_arr == -1)
    eps = np.finfo(np.float32).eps
    # Prepare images_proc
    images_proc = images.squeeze(1).astype(np.float64)  # (N, H_var, 128)
    if use_log:
        images_proc = np.log10(images_proc + eps)
    
    base = images_proc[baseline_mask]  # (B, H_var, 128)
    if robust:
        mu0_map = np.median(base, axis=0)  # (H_var, 128)
    else:
        mu0_map = base.mean(axis=0)
    # %CBV computation
    if use_log:
        ratio = np.power(10.0, images_proc - mu0_map)
        cbv_pct = 100.0 * (ratio - 1.0)
    else:
        denom = np.where(mu0_map != 0, mu0_map, eps)
        cbv_pct = 100.0 * (images_proc - mu0_map) / denom
    cbv_pct = cbv_pct.astype(np.float32)  # (N, H_var, 128)
    # Exclude baseline frames
    data_mask = ~baseline_mask
    cbv_data = cbv_pct[data_mask]  # (M, H_var, 128)
    print(f"CBV range: min={cbv_data.min():.6f}, max={cbv_data.max():.6f}")
    return cbv_data, labels_arr[data_mask]

# Function to clip the %CBV data to the bottom and top percentiles --> this is to remove outliers
def clip(cbv_data, bottom, top):
    if cbv_data.size > 0:  # Avoid empty
        flattened = cbv_data.flatten()  # 1D: M * H_var * 128 values
        p_low = np.percentile(flattened, bottom)   # 5th percentile (lower clip)
        p_high = np.percentile(flattened, top) # 95th percentile (upper clip)
        # Clip to [p_low, p_high]
        cbv_clipped = np.clip(cbv_data, p_low, p_high)
    return cbv_clipped

# Function to interpolate the %CBV data to the target size --> this is to resize the %CBV data to the target size
def tensor_interpolate(cbv_data, target_size = 112):
    cbv_tensor = torch.from_numpy(cbv_data).unsqueeze(1)  # (M, 1, H_var, 128)
    resized_cbv = F.interpolate(
        cbv_tensor, 
        size=(target_size, target_size),  # (H_out, W_out)
        mode='bilinear', 
        align_corners=False
    )
    return resized_cbv

# Function to apply a high pass filter to the %CBV data --> this is to remove low frequency noise
def high_pass(cbv_data, frame_rate = 2.5):
    if cbv_data.size > 0:
        nyquist = frame_rate / 2
        cutoff = 0.05 / nyquist  # Normalized cutoff frequency
        order = 4
        b, a = signal.butter(order, cutoff, btype='high', analog=False)
        cbv_data = signal.filtfilt(b, a, cbv_data, axis=0)  # Apply along time axis
    return cbv_data

# Function to get the height and width of the %CBV data --> this is to resize the %CBV data to the target size
def height_width_matfile(mat_files):
    heights = []
    widths = []
    for mat_path in mat_files:
        print(f"Verifying file: {mat_path}")
        mat_data = scipy.io.loadmat(mat_path)
        doppler_data = mat_data['Doppler']
        doppler_data = doppler_data[np.newaxis, :, :, :]  # Add dim for channel-like
        images = np.transpose(doppler_data, (3, 0, 2, 1)).astype(np.float32)  # (N, 1, H_var, 128)
        h, w = images.shape[2], images.shape[3]
        heights.append(h)
        widths.append(w)
        print(f"  - Shape: {images.shape} (H={h}, W={w})")

    return heights, widths

# Function to split the data into train and test sets --> this is to split the data into train and test sets
def acq_wise_split(big_acquisition_indices,seed=42,split=0.8):
    # Split acquisition-wise into train and test (80% train, 20% test)
    unique_acqs = torch.unique(big_acquisition_indices)
    num_acq = len(unique_acqs)
    print(f"Number of acquisitions: {num_acq}")

    # Set random seed for reproducibility
    random.seed(seed)
    acq_list = list(unique_acqs.numpy())  # Convert to list for shuffling
    random.shuffle(acq_list)

    # Split: approximately 80/20
    train_size = int(split * num_acq)
    train_acqs = acq_list[:train_size]
    test_acqs = acq_list[train_size:]

    print(f"Train acquisitions: {len(train_acqs)}")
    print(f"Test acquisitions: {len(test_acqs)}")

    # Create masks
    train_mask = torch.isin(big_acquisition_indices, torch.tensor(train_acqs))
    test_mask = torch.isin(big_acquisition_indices, torch.tensor(test_acqs))
    return train_mask, test_mask

import torch

# Function to split the data into train and test sets --> this is to split the data into train and test sets
def mid_split_whole(big_acquisition_indices, split=0.8):
    # Apply middle split (40% train, 20% test, 40% train) within each acquisition
    # on the concatenated dataset using big_acquisition_indices to group frames.
    unique_acqs = torch.unique(big_acquisition_indices)
    num_acq = len(unique_acqs)
    print(f"Number of acquisitions: {num_acq}")

    total_N = len(big_acquisition_indices)
    train_indices = []
    test_indices = []

    for acq in unique_acqs:
        mask = (big_acquisition_indices == acq)
        sub_indices = torch.where(mask)[0].tolist()  # Get global indices for this acq
        sub_N = len(sub_indices)

        if sub_N == 0:
            continue

        test_size = int((1 - split) * sub_N)
        train_size_per_side = (sub_N - test_size) // 2
        test_start = train_size_per_side
        test_end = test_start + test_size

        # Adjust if necessary to handle odd sizes
        # Note: The last train part gets any remainder due to integer division

        sub_train = sub_indices[0:train_size_per_side] + sub_indices[test_end:]
        sub_test = sub_indices[test_start:test_end]

        train_indices.extend(sub_train)
        test_indices.extend(sub_test)

    # Create masks
    train_mask = torch.zeros(total_N, dtype=torch.bool)
    test_mask = torch.zeros(total_N, dtype=torch.bool)

    train_mask[torch.tensor(train_indices)] = True
    test_mask[torch.tensor(test_indices)] = True

    print(f"Train frames: {train_mask.sum().item()}")
    print(f"Test frames: {test_mask.sum().item()}")

    return train_mask, test_mask

# Function to shave the data off the start and end --> this is to remove the start and end of the data
def data_shave(images, labels_arr, shave=0):
    # Shave off from start and end 
    M = images.shape[0]
    shave = int(shave * M)
    truncated_cbv = images[shave : M - shave]
    truncated_labels = labels_arr[shave : M - shave]
    return truncated_cbv, truncated_labels

# Function to add shaded regions for contiguous segments of the same label --> this is to add shaded regions to the plot
def add_label_shading(ax, labels, colors_dict, alpha=0.15):
    """Add shaded regions for contiguous segments of the same label."""
    if len(labels) == 0:
        return
    
    # Find indices where label changes
    diff = np.diff(labels)
    change_indices = np.where(diff != 0)[0] + 1
    starts = np.concatenate(([0], change_indices))
    ends = np.concatenate((change_indices, [len(labels)]))
    
    # Shade each segment
    for start, end in zip(starts, ends):
        if start < end:
            label = labels[start]
            color = colors_dict.get(label, 'lightyellow')  # Fallback color
            ax.axvspan(start, end, color=color, alpha=alpha, zorder=0)

# Function to plot the %CBV mean for a specific acquisition --> this is to plot the %CBV mean for a specific acquisition
def plot_cbv_mean(acqs, images, labels, ACQ_INDICE):
    # Create mask for the specific acquisition
    acq_mask = (acqs == ACQ_INDICE)
    # Compute CBV time series for the selected acquisition
    cbv_ts = []
    for i in np.where(acq_mask)[0]:  # Indices where acq_mask is True
        frame_mean = images[i].squeeze().mean().item()  # Mean over HxW
        cbv_ts.append(frame_mean)
    label_colors = { -1: 'lightblue', 0: 'red', 1: 'lightgreen' }
    # Get labels for the selected acquisition
    acq_labels = np.asarray(labels[acq_mask])

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), constrained_layout=True)
    # Add label shading (behind the line)
    add_label_shading(ax, acq_labels, label_colors)
    # CBV% plot
    ax.plot(cbv_ts, color='tab:red', lw=1.0, label='CBV%')
    ax.set_ylabel('CBV% (vs baseline)')
    ax.set_xlabel('Frame')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    plt.show()

# Function to plot the %CBV samples --> this is to plot the %CBV samples
def plot_cbv_samples(cbv_tensor, labels, start=1, step=5, n=10, cmap='inferno',
                     vmin=None, vmax=None, brain_roi=None):
    """
    Plot n frames of %CBV starting at `start`, stepping by `step`.
    - cbv_tensor: torch.Tensor (T,1,H,W) or np.ndarray (T,H,W)
    - brain_roi (optional): (H,W) mask; if provided, percentiles for vmin/vmax
      are computed only inside ROI (ignores zeros outside).
    """
    # Ensure NumPy (T,H,W)
    if isinstance(cbv_tensor, torch.Tensor):
        X = cbv_tensor.detach().cpu().numpy()
    else:
        X = np.asarray(cbv_tensor)

    y = np.asarray(labels)

    if X.ndim == 4 and X.shape[1] == 1:  # (T,1,H,W) -> (T,H,W)
        X = X[:, 0]
    if X.ndim != 3:
        raise ValueError(f"Expected (T,H,W) or (T,1,H,W); got {X.shape}")

    T, H, W = X.shape
    idxs = [start + i*step for i in range(n)]
    idxs = [i for i in idxs if 0 <= i < T]
    if not idxs:
        raise ValueError("No valid indices to plot (check start/step/n vs T).")

    # Grid (2 rows x 5 cols for 10 frames)
    cols = 5
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 3.2*rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    last_im = None
    for ax, i in zip(axes, idxs):
        last_im = ax.imshow(X[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Frame {i}, Label: {y[i]}")
        ax.axis('off')

    # Hide any unused subplots
    for ax in axes[len(idxs):]:
        ax.axis('off')

    # One shared colorbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[:len(idxs)], shrink=0.8, fraction=0.03, pad=0.02)
        cbar.set_label('%CBV')

    plt.show()

# Function to compute the label distribution --> this is to compute the label distribution # no use in  baseline
def pause_work_distrib(labels):
    # Compute label distribution (excluding -1, which isn't used in valid windows)
    label_counts = np.bincount(labels, minlength=2)  # Counts for 0, 1
    total_valid = sum(label_counts)

    # Print for verification
    for label, count in enumerate(label_counts):
        percentage = (count / total_valid * 100) if total_valid > 0 else 0
        print(f"Label {label}: {count} patches ({percentage:.2f}%)")

# Function to normalize the data to [0, 1] --> this is to normalize the data to [0, 1]
def norm(data):
    """
    Normalize image data to [0, 1] using min-max normalization.
    
    Args:
        data: Input tensor or array (e.g., PyTorch tensor or NumPy array) of image data.
              Expected shape: [M, H, W] or similar (M frames, H height, W width).
    
    Returns:
        normalized_data: Tensor/array of the same shape, normalized to [0, 1].
    """
    # Convert to NumPy if input is PyTorch tensor for consistent handling
    is_torch = isinstance(data, torch.Tensor)
    if is_torch:
        data_np = data.cpu().numpy()
    else:
        data_np = data

    # Check for empty or invalid data
    if data_np.size == 0:
        print("Warning: Empty data, returning zeros.")
        return torch.zeros_like(data) if is_torch else np.zeros_like(data)

    # Compute min and max
    data_min = data_np.min()
    data_max = data_np.max()

    # Handle case where max equals min (constant data)
    if data_max == data_min:
        print("Warning: Data is constant, returning zeros.")
        return torch.zeros_like(data) if is_torch else np.zeros_like(data)

    # Normalize to [0, 1]
    normalized_data = (data_np - data_min) / (data_max - data_min)

    # Clip to [0, 1] to handle numerical precision issues
    normalized_data = np.clip(normalized_data, 0, 1)

    # Convert back to PyTorch tensor if input was a tensor
    if is_torch:
        normalized_data = torch.from_numpy(normalized_data).to(data.dtype)

    # Debugging: Print min and max of normalized data
    print(f"Normalized range: min={normalized_data.min():.6f}, max={normalized_data.max():.6f}")

    return normalized_data

# Function to compute the frame difference --> this is to compute the difference in haemodynamic response from frame to frame
def frame_diff(images: np.ndarray, mode: str = "window", window: int = 8) -> np.ndarray:
    """
    Compute frame-to-frame hemodynamic change (differential imaging).
    
    Parameters
    ----------
    images : np.ndarray
        Shape (T, 1, H, W) - raw Doppler intensity
    mode : str
        "consecutive": ΔI_t = I_t - I_{t-1}
        "window": ΔI_t = I_t - mean(I_{t-w:t})
    window : int
        Number of previous frames for window baseline
    
    Returns
    -------
    diff : np.ndarray
        Same shape as input
    """
    images = images.astype(np.float32)
    T = images.shape[0]
    diff = np.zeros_like(images)

    if mode == "consecutive":
        diff[1:] = images[1:] - images[:-1]
        diff[0] = images[0]  # or 0
    elif mode == "window":
        for t in range(T):
            start = max(0, t - window)
            baseline = np.mean(images[start:t], axis=0, keepdims=True) if t > 0 else images[0:1]
            diff[t] = images[t:t+1] - baseline
    else:
        raise ValueError("mode must be 'consecutive' or 'window'")
    
    return diff


import torch
import torch.nn.functional as F
# Function to pad or crop the %CBV data to the target size --> this is to pad or crop the %CBV data to the target size
def tensor_pad_or_crop(cbv_data, target_size: int = 112):
    """
    Convert a fUS CBV acquisition to (N, 1, target_size, target_size) by 
    padding with zeros at the bottom (if H < target_size) or 
    cropping the bottom part (if H > target_size).
    No interpolation → perfect preservation of original values.
    
    Parameters
    ----------
    cbv_data : np.ndarray or torch.Tensor
        Shape (N_images, H_var, 128) or (N_images, 1, H_var, 128)
    target_size : int
        Desired height and width (default 112 for VideoMAE-small)
        
    Returns
    -------
    torch.Tensor
        Shape (N_images, 1, target_size, target_size), float32
    """
    if not isinstance(cbv_data, torch.Tensor):
        cbv_tensor = torch.from_numpy(cbv_data)
    else:
        cbv_tensor = cbv_data
        
    # Ensure channel dimension exists: (N, H, 128) → (N, 1, H, 128)
    if cbv_tensor.dim() == 3:
        cbv_tensor = cbv_tensor.unsqueeze(1)
    # Now shape is (N, 1, H_var, 128)

    N, C, H, W = cbv_tensor.shape
    target = target_size
    
    if H == target:
        # Already perfect size in height
        if W != target:
            # This should never happen in your case (W is always 128), but we fix it anyway
            cbv_tensor = F.pad(cbv_tensor, (0, target - W, 0, 0)) if W < target else cbv_tensor[:, :, :, :target]
        return cbv_tensor.float()
    
    # Height handling
    if H < target:
        # Pad bottom with zeros
        pad_bottom = target - H
        cbv_tensor = F.pad(cbv_tensor, (0, 0, 0, pad_bottom), mode='constant', value=0)
    else:
        # Crop bottom part
        cbv_tensor = cbv_tensor[:, :, :target, :]
    
    # Width handling (in case someone changes the probe)
    if W < target:
        pad_right = target - W
        cbv_tensor = F.pad(cbv_tensor, (0, pad_right, 0, 0), mode='constant', value=0)
    elif W > target:
        cbv_tensor = cbv_tensor[:, :, :, :target]
    
    return cbv_tensor.float()

import numpy as np

import numpy as np
# Function to pad or crop the %CBV data to the target size --> this is to pad or crop the %CBV data to the target size
def np_pad_or_crop_to_square(cbv_data, target_size: int = 112):
    """
    Same as before, but preserves/creates a leading channel dimension.
    
    Input:
        (N, H_var, 128)          → no channel
        (N, 1, H_var, 128)       → already has channel
    
    Output:
        (N, 1, target_size, target_size)   # always 4D with channel=1
    """
    # ------------------------------------------------------------------
    # 1. Ensure we have 4 dimensions (N, C, H, W) with C=1
    # ------------------------------------------------------------------
    if cbv_data.ndim == 3:
        # (N, H, W) → add channel dim
        data = cbv_data[:, np.newaxis, :, :]        # (N, 1, H, W)
    elif cbv_data.ndim == 4:
        if cbv_data.shape[1] != 1:
            raise ValueError("Channel dimension exists but is not 1")
        data = cbv_data
    else:
        raise ValueError(f"Expected 3 or 4 dims, got {cbv_data.ndim}")

    N, C, H, W = data.shape
    target = target_size

    # ------------------------------------------------------------------
    # 2. Pad or crop height
    # ------------------------------------------------------------------
    if H < target:
        pad_bottom = target - H
        pad_width = ((0, 0), (0, 0), (0, pad_bottom), (0, 0))  # (N, C, H, W)
        data = np.pad(data, pad_width, mode='constant', constant_values=0)
    elif H > target:
        data = data[:, :, :target, :]

    # ------------------------------------------------------------------
    # 3. Pad or crop width (your raw data is always 128 px wide)
    # ------------------------------------------------------------------
    current_h, current_w = data.shape[2], data.shape[3]
    if current_w < target:
        pad_right = target - current_w
        pad_width = ((0, 0), (0, 0), (0, 0), (0, pad_right))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)
    elif current_w > target:
        data = data[:, :, :, :target]
    print("Function pad shape: ", np.shape(data))
    # Final guaranteed shape: (N, 1, target_size, target_size)
    return data

# Function to create the ROI --> this is to create the ROI # not needed for my project
def create_roi(images, file_idx):
    # Load the first frame from cbv_tensor_dataset
    frame = images[0].squeeze()  # Shape: (height, width) or (channels, height, width)
    if frame.ndim == 3:  # If channels exist, take first or mean
        frame = frame[0] if frame.shape[0] < frame.shape[-1] else np.mean(frame, axis=0)

    # Set up the figure
    fig, ax = plt.subplots()
    ax.imshow(frame, cmap='gray')  # Use grayscale colormap; adjust if needed
    ax.set_title('Click points to define ROI (right-click or Enter to finish)')

    # Collect points interactively
    print("Click points to define the ROI. Right-click or press Enter to close the polygon.")
    points = plt.ginput(n=-1, timeout=-1, show_clicks=True)  # n=-1 allows unlimited points; timeout=-1 waits indefinitely
    plt.close()  # Close the plot after selection

    # Convert points to numpy array
    points = np.array(points)  # Shape: (num_points, 2) for (x, y) coordinates

    # Create a closed polygon Path
    # Append first point to close the polygon
    points = np.vstack([points, points[0]])
    path = Path(points)

    # Generate boolean mask for the ROI
    height, width = frame.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x, y = x.flatten(), y.flatten()
    points_grid = np.vstack((x, y)).T
    mask = path.contains_points(points_grid)
    mask = mask.reshape((height, width))
    
    # Save or use the mask
    # mask is a boolean numpy array: True inside ROI, False outside
    # You can save it or apply it to your tensor dataset
    np.save(f'roi_{file_idx}.npy', mask)  # Save for later use


import numpy as np

# Function to compute the %CBV change relative to baseline, using only ROI pixels --> this is to compute the %CBV change relative to baseline, using only ROI pixels
def delta_cbv_roi(images, labels_arr, roi_mask, use_log=False, robust=True):
    """
    Compute %CBV change relative to baseline, using only ROI pixels.
    
    Parameters
    ----------
    images : np.ndarray
        Shape (N, 1, H, W) or (N, H, W) – your fUS frames (raw Doppler or power)
    labels_arr : np.ndarray
        Shape (N,) with -1 = baseline, 0 = success, 1 = mistake
    roi_mask : np.ndarray bool
        Shape (H, W) – True inside the region of interest
    use_log : bool
        If True → log-ratio method (more robust to outliers)
    robust : bool
        If True → use median baseline instead of mean
    
    Returns
    -------
    cbv_data : np.ndarray
        Shape (M, H, W), float32 – %CBV change, zero outside ROI, only non-baseline frames
    labels_filtered : np.ndarray
        Shape (M,) – corresponding behavioral labels (baseline frames removed)
    """
    eps = np.finfo(np.float32).eps
    
    # ------------------------------------------------------------------
    # 1. Ensure (N, H, W) and float64 for precision
    # ------------------------------------------------------------------
    if images.ndim == 4:
        images = images.squeeze(axis=1)          # (N,1,H,W) → (N,H,W)
    images = images.astype(np.float64)

    H, W = images.shape[1], images.shape[2]
    if roi_mask.shape != (H, W):
        raise ValueError(f"Mask shape {roi_mask.shape} doesn't match image spatial dims {(H,W)}")

    # Expand mask for broadcasting: (1, H, W)
    mask = roi_mask[np.newaxis, :, :].astype(bool)

    # Mask non-ROI → NaN so they don't affect statistics
    images_masked = np.where(mask, images, np.nan)

    # ------------------------------------------------------------------
    # 2. Compute baseline from ROI pixels only
    # ------------------------------------------------------------------
    baseline_idx = (labels_arr == -1)
    baseline_frames = images_masked[baseline_idx]      # (B, H, W)

    if baseline_frames.size == 0:
        raise ValueError("No baseline frames found (label == -1)")

    if robust:
        baseline_map = np.nanmedian(baseline_frames, axis=0, keepdims=True)  # (1,H,W)
    else:
        baseline_map = np.nanmean(baseline_frames, axis=0, keepdims=True)

    # Optional log transform
    if use_log:
        images_masked = np.log10(images_masked + eps)
        baseline_map = np.log10(baseline_map + eps)

    # ------------------------------------------------------------------
    # 3. Compute ΔCBV
    # ------------------------------------------------------------------
    if use_log:
        # Log-ratio → convert back to linear ratio
        ratio = 10.0 ** (images_masked - baseline_map)
        cbv_pct = 100.0 * (ratio - 1.0)
    else:
        denom = np.where(baseline_map == 0, eps, baseline_map)
        cbv_pct = 100.0 * (images_masked - baseline_map) / denom

    cbv_pct = cbv_pct.astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Remove baseline frames + zero out non-ROI
    # ------------------------------------------------------------------
    non_baseline_idx = ~baseline_idx
    cbv_data = cbv_pct[non_baseline_idx]                    # (M, H, W)
    cbv_data[:, ~roi_mask] = 0.0                # zero outside ROI

    labels_filtered = labels_arr[non_baseline_idx]

    # ------------------------------------------------------------------
    # 5. Quick sanity print
    # ------------------------------------------------------------------
    roi_min, roi_max = np.nanmin(cbv_pct[non_baseline_idx]), np.nanmax(cbv_pct[non_baseline_idx])
    print(f"Acquisition ΔCBV (in ROI) → min: {roi_min:+.3f}%, max: {roi_max:+.3f}% | "
          f"Frames kept: {len(labels_filtered)} | ROI pixels: {roi_mask.sum()}")

    return cbv_data, labels_filtered

# Function to normalize the %CBV data using only ROI pixels --> this is to normalize the %CBV data using only ROI pixels
import numpy as np

def normalize_cbv_in_roi(cbv_data, roi_mask, method="robust", eps=1e-8):
    """
    Normalize CBV data using statistics computed EXCLUSIVELY from ROI pixels.
    Pixels outside the ROI remain exactly 0.0 (no change).
    
    Parameters
    ----------
    cbv_data : np.ndarray
        Shape (M, H, W) – ΔCBV frames (float32), with 0s outside ROI
    roi_mask : np.ndarray bool
        Shape (H, W) – True inside ROI
    method : str
        - "zscore"  → zero-mean, unit-variance (recommended for MAEs/CNNs)
        - "minmax"  → scale to [0, 1] or [-1, 1] depending on sign
        - "robust"  → (x - median) / IQR  (very robust to outliers)
    eps : float
        Small constant to avoid division by zero
    
    Returns
    -------
    cbv_normalized : np.ndarray
        Same shape as input, normalized only in ROI, 0 elsewhere
    """
    if cbv_data.shape[1:] != roi_mask.shape:
        raise ValueError(f"cbv_data spatial dims {cbv_data.shape[1:]} != mask {roi_mask.shape}")

    # Extract only ROI pixels across all frames → (M_roi_pixels,)
    mask = roi_mask.astype(bool)                 # (H, W)
    roi_values = cbv_data[:, mask]                # ← correct indexing

    if roi_values.size == 0:
        raise ValueError("ROI mask is empty!")

    cbv_norm = np.zeros_like(cbv_data)

    if method == "zscore":
        mean = roi_values.mean()
        std  = roi_values.std()
        if std < eps:
            print("Warning: ROI std very small → using min-max fallback")
            cbv_norm = np.where(mask, (cbv_data - cbv_data.min()) / (cbv_data.max() - cbv_data.min() + eps), 0.0)
        else:
            cbv_norm = np.where(mask, (cbv_data - mean) / (std + eps), 0.0)

    elif method == "minmax":
        vmin, vmax = roi_values.min(), roi_values.max()
        if (vmax - vmin) < eps:
            cbv_norm = np.where(mask, 0.0, 0.0)
        else:
            cbv_norm = np.where(mask, (cbv_data - vmin) / (vmax - vmin + eps), 0.0)

    elif method == "robust":
        median = np.median(roi_values)
        iqr = np.percentile(roi_values, 75) - np.percentile(roi_values, 25)
        scale = iqr / 1.349  # approximate std for normal dist
        if scale < eps:
            scale = 1.0
        cbv_norm = np.where(mask, (cbv_data - median) / (scale + eps), 0.0)

    else:
        raise ValueError("method must be 'zscore', 'minmax', or 'robust'")

    print(f"ROI normalization ({method}): "
          f"mean={roi_values.mean():.4f}, std={roi_values.std():.4f}, "
          f"min={roi_values.min():.3f}, max={roi_values.max():.3f}")


    return cbv_norm.astype(np.float32)



import numpy as np
from scipy.ndimage import uniform_filter


import numpy as np
from scipy.ndimage import uniform_filter
# Function to compute the %CBV change relative to baseline, using only ROI pixels --> this is to compute the %CBV change relative to baseline, using only ROI pixels
def delta_cbv_roi_adaptive(
    images,
    labels_arr,
    roi_mask,
    window_sec=20.0,
    acquisition_rate_hz=2.0,
    use_log=False,
    robust=True,
    spatial_filter_radius=2,
    eps=np.finfo(np.float32).eps
):
    """
    Adaptive ΔCBV with rolling z-score, ROI-only statistics, and output shape (M, 1, H, W)
    """
    window_frames = int(window_sec * acquisition_rate_hz)
    if window_frames < 10:
        raise ValueError("Window too small – increase window_sec or check rate")

    # ------------------------------------------------------------------
    # 1. Prep images: (N, H, W) float64
    # ------------------------------------------------------------------
    if images.ndim == 4:
        images = images.squeeze(axis=1)           # (N,1,H,W) → (N,H,W)
    images = images.astype(np.float64)
    N, H, W = images.shape

    if roi_mask.shape != (H, W):
        raise ValueError(f"Mask {roi_mask.shape} != image dims {(H,W)}")

    mask = roi_mask[np.newaxis, :, :].astype(bool)   # (1,H,W)

    if use_log:
        images = np.log10(images + eps)

    # ------------------------------------------------------------------
    # 2. Rolling z-score using ROI only
    # ------------------------------------------------------------------
    preprocessed = np.zeros((N, H, W), dtype=np.float32)
    
    start_idx = window_frames

    for i in range(start_idx, N):
        buffer = images[i - window_frames : i]                    # (win, H, W)
        buffer_masked = np.where(mask, buffer, np.nan)

        if robust:
            mu = np.nanmedian(buffer_masked, axis=0)
        else:
            mu = np.nanmean(buffer_masked, axis=0)

        sigma = np.nanstd(buffer_masked, axis=0) + eps

        current_masked = np.where(roi_mask, images[i], np.nan)
        z_frame = (current_masked - mu) / sigma
        '''
        if spatial_filter_radius is not None and spatial_filter_radius > 0:
            size = 2 * spatial_filter_radius + 1
            z_frame = uniform_filter(z_frame, size=size, mode='reflect')
        '''
        preprocessed[i] = np.where(roi_mask, z_frame, 0.0)

    # ------------------------------------------------------------------
    # 3. Keep only non-baseline frames after warm-up
    # ------------------------------------------------------------------
    valid_mask = (np.arange(N) >= start_idx) & (labels_arr != -1)
    cbv_data = preprocessed[valid_mask]               # (M, H, W)
    labels_filtered = labels_arr[valid_mask]

    # ------------------------------------------------------------------
    # 4. Add channel dimension back → (M, 1, H, W)
    # ------------------------------------------------------------------
    # cbv_data = cbv_data[:, np.newaxis, :, :].astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Print stats
    # ------------------------------------------------------------------
    if len(cbv_data) == 0:
        raise ValueError("No valid frames after warm-up and baseline removal")
    
    roi_vals = cbv_data[np.tile(roi_mask, (len(cbv_data), 1, 1))]
    roi_min, roi_max = np.min(roi_vals), np.max(roi_vals)
    print(f"Adaptive ΔCBV (window={window_sec}s={window_frames}frames, ROI only) "
          f"→ min: {roi_min:+.3f}, max: {roi_max:+.3f} | "
          f"Frames kept: {len(labels_filtered)} | ROI pixels: {roi_mask.sum()} | "
          f"Discarded first {start_idx} frames (warm-up)")

    return cbv_data, labels_filtered


import numpy as np
from scipy.ndimage import convolve
# Function to create the pillbox kernel --> this is to create the pillbox kernel
def create_pillbox_kernel(radius: int):
    """
    Create a normalized 2D circular (pillbox) kernel.
    """
    if radius < 0:
        raise ValueError("Radius must be >= 0")
    if radius == 0:
        return np.ones((1, 1), dtype=np.float32)

    diameter = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    kernel = (xx**2 + yy**2 <= radius**2).astype(np.float32)
    kernel /= kernel.sum()  # normalize
    return kernel  # shape (diameter, diameter)

# Function to apply the pillbox filter --> this is to apply the pillbox filter
def pillbox_filter(data: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Apply isotropic circular pillbox (uniform disk) smoothing.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (M, H, W) – your CBV or z-scored frames (e.g. from delta_cbv_roi_adaptive)
    radius : int
        Radius of the disk in pixels (e.g. 2 → 5×5 circular kernel)
    
    Returns
    -------
    filtered : np.ndarray
        Same shape (M, H, W), float32, smoothly filtered
    """
    if data.ndim != 3:
        raise ValueError(f"Expected (M, H, W), got shape {data.shape}")

    if radius == 0:
        return data.astype(np.float32).copy()

    kernel = create_pillbox_kernel(radius)                     # (d, d)
    kernel = kernel.reshape(1, 1, kernel.shape[0], kernel.shape[1])  # (1,1,d,d)

    # Convolve across the batch (M frames)
    filtered = convolve(data[:, np.newaxis, :, :], kernel, mode='reflect')
    filtered = filtered.squeeze(1)  # remove channel dim → (M, H, W)

    print(f"Pillbox filter applied | radius={radius}px | "
          f"kernel={2*radius+1}×{2*radius+1} ({kernel.sum()} pixels) | frames={data.shape[0]}")

    return filtered.astype(np.float32)

import numpy as np
from sklearn.decomposition import PCA
# Function to perform PCA-based denoising --> this is to perform PCA-based denoising
def pca_denoise(images, n_components=None, var_keep=0.70):
    """
    Perform PCA-based denoising on a 3D array (T, H, W).

    Parameters
    ----------
    images : np.ndarray
        Data array of shape (n_frames, H, W).
    n_components : int or None
        Number of PCA components to retain. If None,
        the number will be chosen to keep `var_keep` of total variance.
    var_keep : float
        Fraction of variance to retain when n_components is None.

    Returns
    -------
    images_denoised : np.ndarray
        Data reconstructed after PCA denoising.
    pca : sklearn.decomposition.PCA
        The fitted PCA object (useful for diagnostics).
    """
    T, H, W = images.shape
    # Flatten spatial dimensions
    X = images.reshape(T, H * W)

    # Fit PCA across frames (time is samples)
    pca = PCA()
    pca.fit(X)  # Fit on full to get cumvar

    # Determine n_components
    available_components = pca.n_components_
    if n_components is None:
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.searchsorted(cumvar, var_keep) + 1
        n_components = min(n_components, available_components)

    # Transform with full, then subset
    X_trans = pca.transform(X)  # (T, available)

    # Manual inverse with subset
    X_subset = X_trans[:, :n_components]  # (T, n_components)
    components_subset = pca.components_[:n_components, :]  # (n_components, H*W)
    explained_variance_subset = pca.explained_variance_[:n_components]

    # Inverse (whiten=False, default)
    X_rec = X_subset @ components_subset + pca.mean_

    # Reshape back to (T, H, W)
    images_denoised = X_rec.reshape(T, H, W)

    return images_denoised, pca
# Function to get the ROI mask --> this is to get the ROI mask
def get_or_create_roi_mask(images, file_idx):
    """
    Returns the ROI mask for a given acquisition.
    - If roi_{file_idx}.npy already exists → loads it
    - Else → launches the interactive drawing and then loads it
    """
    mask_path = f'roi_{file_idx}.npy'
    
    if os.path.exists(mask_path):
        print(f"ROI mask found → loading {mask_path}")
        mask = np.load(mask_path)
    else:
        print(f"No ROI mask found → starting interactive drawing for acquisition {file_idx}")
        create_roi(images, file_idx)          # this saves the .npy file
        mask = np.load(mask_path)
        print(f"Mask created and loaded → {mask_path}")
    
    return mask


