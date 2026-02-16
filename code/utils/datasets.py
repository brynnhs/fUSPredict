from collections import OrderedDict
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import helper_functions as hf


class FUSForecastWindowDataset(Dataset):
    """
    Per-acquisition sliding-window forecasting dataset.
    Returns:
      - context: [W, C, H, W]
      - target:  [K, C, H, W]
    If return_meta=True:
      - (context, target, {"subject", "path", "end_ctx"})
    """

    def __init__(
        self,
        manifest_path=None,
        acq_paths=None,
        split=None,
        window_size=8,
        pred_horizon=1,
        stride=1,
        frames_key="frames",
        labels_key=None,
        mask_key=None,
        exclude_label=-1,
        lru_cache_size=2,
        target_size=112,
        return_meta=False,
    ):

        if window_size <= 0 or pred_horizon <= 0 or stride <= 0:
            raise ValueError("window_size, pred_horizon, and stride must all be > 0")
        if manifest_path is None and acq_paths is None:
            raise ValueError("Provide either manifest_path or acq_paths")
        self.window_size = int(window_size)
        self.pred_horizon = int(pred_horizon)
        self.stride = int(stride)
        self.frames_key = frames_key
        self.labels_key = labels_key
        self.mask_key = mask_key
        self.exclude_label = exclude_label
        self.lru_cache_size = int(max(1, lru_cache_size))
        self.target_size = int(target_size)
        self.return_meta = bool(return_meta)
        self.acq_paths, self.acq_subjects = self._resolve_inputs(manifest_path, acq_paths, split)
        for p in self.acq_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing acquisition file: {p}")
        self._acq_cache = OrderedDict()
        self.index_map = []
        self.acq_meta = []
        self.expected_chw = None
        self._build_index_map()
        if len(self.index_map) == 0:
            raise RuntimeError("No valid windows found. Check window/pred/stride and exclusion rules.")

    def _resolve_inputs(self, manifest_path, acq_paths, split):
        path_to_subject = {}
        if manifest_path is not None:
            with open(manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            # Optional metadata map: meta[path] -> {subject: ...}
            meta = m.get("meta", {})
            if isinstance(meta, dict):
                for p, info in meta.items():
                    if isinstance(info, dict) and info.get("subject") is not None:
                        path_to_subject[str(p)] = str(info["subject"])
            # Optional acquisitions list: [{path, subject}, ...]
            acqs = m.get("acquisitions", [])
            if isinstance(acqs, list):
                for item in acqs:
                    if isinstance(item, dict) and "path" in item and "subject" in item:
                        path_to_subject[str(item["path"])] = str(item["subject"])
            if split is None:
                paths = list(m.get("train", [])) + list(m.get("test", []))
            else:
                if split not in ("train", "test"):
                    raise ValueError("split must be 'train' or 'test'")
                paths = list(m.get(split, []))
        else:
            paths = list(acq_paths)
        if not paths:
            raise ValueError("No acquisition files provided")
        paths = [str(p) for p in paths]
        subjects = [path_to_subject.get(p, Path(p).parent.name) for p in paths]
        return paths, subjects

    def _get_invalid_mask(self, z, Tn, path):
        invalid = np.zeros(Tn, dtype=bool)
        if self.labels_key is not None and self.labels_key in z:
            labels = np.asarray(z[self.labels_key]).squeeze()
            if labels.shape[0] != Tn:
                raise ValueError(f"{path}: labels length {labels.shape[0]} != T {Tn}")
            invalid |= (labels == self.exclude_label)
        if self.mask_key is not None and self.mask_key in z:
            mask = np.asarray(z[self.mask_key]).squeeze()
            if mask.shape[0] != Tn:
                raise ValueError(f"{path}: mask length {mask.shape[0]} != T {Tn}")
            invalid |= mask.astype(bool)
        return invalid

    def _normalize_frames(self, frames, path):
        if frames.ndim == 3:
            frames = frames[:, np.newaxis, :, :]
        if frames.ndim != 4:
            raise ValueError(f"{path}: frames must be [T,H,W] or [T,C,H,W], got {frames.shape}")
        if frames.shape[1] != 1:
            raise ValueError(f"{path}: np_pad_or_crop_to_square expects channel=1, got C={frames.shape[1]}")
        frames = hf.np_pad_or_crop_to_square(frames, target_size=self.target_size).astype(np.float32, copy=False)
        return frames

    def _build_index_map(self):
        for acq_idx, path in enumerate(self.acq_paths):
            with np.load(path, allow_pickle=False) as z:
                if self.frames_key not in z:
                    raise KeyError(f"{path} missing frames key '{self.frames_key}'")
                frames = self._normalize_frames(z[self.frames_key], path)
                Tn, Cn, Hn, Wn = frames.shape
                if self.expected_chw is None:
                    self.expected_chw = (Cn, Hn, Wn)
                elif self.expected_chw != (Cn, Hn, Wn):
                    raise ValueError(
                        f"Inconsistent frame shape across acquisitions: expected {self.expected_chw}, got {(Cn, Hn, Wn)} in {path}"
                    )
                if Tn < self.window_size + self.pred_horizon:
                    self.acq_meta.append({"T": Tn, "shape": (Cn, Hn, Wn), "n_windows": 0})
                    continue
                invalid = self._get_invalid_mask(z, Tn, path)
                n_windows = 0
                for end_ctx in range(self.window_size - 1, Tn - self.pred_horizon, self.stride):
                    start_ctx = end_ctx - self.window_size + 1
                    target_end = end_ctx + self.pred_horizon
                    if start_ctx < 0 or target_end >= Tn:
                        continue
                    if invalid[start_ctx: target_end + 1].any():
                        continue
                    self.index_map.append((acq_idx, end_ctx))
                    n_windows += 1
                self.acq_meta.append({"T": Tn, "shape": (Cn, Hn, Wn), "n_windows": n_windows})

    def __len__(self):
        return len(self.index_map)

    def _load_acquisition(self, acq_idx):
        if acq_idx in self._acq_cache:
            x = self._acq_cache.pop(acq_idx)
            self._acq_cache[acq_idx] = x
            return x
        path = self.acq_paths[acq_idx]
        with np.load(path, allow_pickle=False) as z:
            frames = self._normalize_frames(z[self.frames_key], path)
        _, Cn, Hn, Wn = frames.shape
        if self.expected_chw != (Cn, Hn, Wn):
            raise ValueError(f"{path}: shape changed vs expected {self.expected_chw}, got {(Cn, Hn, Wn)}")
        self._acq_cache[acq_idx] = frames
        if len(self._acq_cache) > self.lru_cache_size:
            self._acq_cache.popitem(last=False)
        return frames

    def __getitem__(self, idx):
        acq_idx, end_ctx = self.index_map[idx]
        frames = self._load_acquisition(acq_idx)
        start_ctx = end_ctx - self.window_size + 1
        target_start = end_ctx + 1
        target_end = end_ctx + self.pred_horizon

        if start_ctx < 0 or target_end >= frames.shape[0]:
            raise IndexError("Computed window is out of bounds")
        context = frames[start_ctx:end_ctx + 1]  # [W, C, H, W]
        target = frames[target_start:target_end + 1]  # [K, C, H, W]

        if context.shape[0] != self.window_size:
            raise RuntimeError(f"Context length mismatch: expected {self.window_size}, got {context.shape[0]}")

        if target.shape[0] != self.pred_horizon:
            raise RuntimeError(f"Target length mismatch: expected {self.pred_horizon}, got {target.shape[0]}")

        x = torch.from_numpy(context)
        y = torch.from_numpy(target)

        if x.dtype != torch.float32 or y.dtype != torch.float32:
            raise TypeError("Dataset outputs must be float32 tensors")

        if self.return_meta:
            info = {
                "subject": self.acq_subjects[acq_idx],
                "path": self.acq_paths[acq_idx],
                "end_ctx": int(end_ctx),
            }
            return x, y, info
        return x, y
