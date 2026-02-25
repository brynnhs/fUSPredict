import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class FUSNPZForecastDataset(Dataset):
    """OpenSTL-compatible dataset for manifest-driven fUS NPZ forecasting."""

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        frames_key: str = "frames",
        pre_seq_length: int = 20,
        aft_seq_length: int = 20,
        stride: int = 1,
        random_window: bool | None = None,
        seed: int = 42,
        skip_short: bool = True,
        lru_cache_size: int = 8,
        return_meta: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'test'}")
        if pre_seq_length <= 0 or aft_seq_length <= 0:
            raise ValueError("pre_seq_length and aft_seq_length must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
        self.split = split
        self.frames_key = frames_key
        self.pre_seq_length = int(pre_seq_length)
        self.aft_seq_length = int(aft_seq_length)
        self.total_len = self.pre_seq_length + self.aft_seq_length
        self.stride = int(stride)
        self.random_window = (
            (split == "train") if random_window is None else bool(random_window)
        )
        self.seed = int(seed)
        self.skip_short = bool(skip_short)
        self.return_meta = bool(return_meta)
        self._rng = random.Random(self.seed)
        self._cache_max = max(1, int(lru_cache_size))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

        with self.manifest_path.open("r", encoding="utf-8") as f:
            manifest: dict[str, Any] = json.load(f)

        if self.frames_key == "frames":
            self.frames_key = str(manifest.get("frames_key", self.frames_key))

        self._meta_by_path = self._build_meta_index(manifest)
        source_paths = self._pick_split_paths(manifest, split)
        self.samples = self._build_sample_index(source_paths)
        if len(self.samples) == 0:
            raise RuntimeError(f"No usable samples found for split='{split}'.")

    def _build_meta_index(self, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        meta = manifest.get("meta", {})
        if isinstance(meta, dict):
            for path_str, info in meta.items():
                if isinstance(info, dict):
                    out[self._resolve_data_path(path_str).as_posix()] = dict(info)
        acquisitions = manifest.get("acquisitions", [])
        if isinstance(acquisitions, list):
            for item in acquisitions:
                if not isinstance(item, dict) or "path" not in item:
                    continue
                key = self._resolve_data_path(item["path"]).as_posix()
                prev = out.get(key, {})
                merged = dict(prev)
                merged.update({k: v for k, v in item.items() if k != "path"})
                out[key] = merged
        return out

    def _pick_split_paths(self, manifest: dict[str, Any], split: str) -> list[str]:
        if split == "train":
            paths = manifest.get("train", [])
        elif split == "val":
            paths = manifest.get("val") or manifest.get("test", [])
        else:
            paths = manifest.get("test") or manifest.get("val", [])
        if not isinstance(paths, list):
            raise ValueError(f"Manifest split '{split}' must be a list.")
        return [self._resolve_data_path(p).as_posix() for p in paths]

    def _resolve_data_path(self, path_str: str | Path) -> Path:
        p = Path(path_str).expanduser()
        if not p.is_absolute():
            p = (self.manifest_path.parent / p).resolve(strict=False)
        else:
            p = p.resolve(strict=False)
        return p

    def _build_sample_index(self, paths: list[str]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for path in sorted(paths):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Missing NPZ file referenced by manifest: {path}"
                )
            with np.load(path, allow_pickle=False) as z:
                if self.frames_key not in z:
                    raise KeyError(f"{path} missing key '{self.frames_key}'")
                frames = self._to_thw(z[self.frames_key], path)
            t = int(frames.shape[0])
            if t < self.total_len:
                if self.skip_short:
                    continue
                raise ValueError(
                    f"{path} has T={t} < required total length {self.total_len}"
                )
            samples.append({"path": path, "T": t})
        return samples

    def _to_thw(self, frames: np.ndarray, path: str) -> np.ndarray:
        arr = np.asarray(frames)

        if arr.ndim == 4:
            squeeze_axes = tuple(i for i, s in enumerate(arr.shape) if s == 1)
            if not squeeze_axes:
                raise ValueError(
                    f"{path}: expected grayscale frames, got shape {arr.shape}"
                )
            arr = np.squeeze(arr, axis=squeeze_axes)

        if arr.ndim != 3:
            raise ValueError(
                f"{path}: expected [T,H,W] or reorderable 3D array, got {arr.shape}"
            )

        time_axis = self._infer_time_axis(arr.shape)
        if time_axis != 0:
            arr = np.moveaxis(arr, time_axis, 0)

        return np.ascontiguousarray(arr.astype(np.float32, copy=False))

    def _infer_time_axis(self, shape: tuple[int, int, int]) -> int:
        scores: list[tuple[float, float, int]] = []
        for axis in range(3):
            spatial = [shape[i] for i in range(3) if i != axis]
            sim = abs(spatial[0] - spatial[1]) / max(spatial[0], spatial[1], 1)
            length_penalty = 0.0 if shape[axis] >= self.total_len else 1.0
            prefer_front = 0 if axis == 0 else 1
            scores.append((length_penalty, sim, prefer_front + axis))
        best = min(range(3), key=lambda i: scores[i])
        return best

    def _load_frames(self, path: str) -> np.ndarray:
        if path in self._cache:
            arr = self._cache.pop(path)
            self._cache[path] = arr
            return arr

        with np.load(path, allow_pickle=False) as z:
            if self.frames_key not in z:
                raise KeyError(f"{path} missing key '{self.frames_key}'")
            arr = self._to_thw(z[self.frames_key], path)
        arr = arr[:, None, :, :]  # [T, 1, H, W]
        self._cache[path] = arr
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return arr

    def _eval_start(self, max_start: int) -> int:
        if max_start <= 0:
            return 0
        # Even seed -> left-aligned, odd seed -> centered.
        return 0 if (self.seed % 2 == 0) else (max_start // 2)

    def _select_start(self, t: int) -> int:
        max_start = t - self.total_len
        if max_start <= 0:
            return 0
        if self.random_window:
            start = self._rng.randint(0, max_start)
            if self.stride > 1:
                start = (start // self.stride) * self.stride
            return min(start, max_start)
        return self._eval_start(max_start)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        path = sample["path"]
        frames = self._load_frames(path)
        t = frames.shape[0]
        start = self._select_start(t)
        stop = start + self.total_len
        clip = frames[start:stop]
        if clip.shape[0] != self.total_len:
            raise RuntimeError(f"Invalid clip length at {path}: got {clip.shape[0]}")

        x = torch.from_numpy(clip[: self.pre_seq_length]).to(dtype=torch.float32)
        y = torch.from_numpy(clip[self.pre_seq_length :]).to(dtype=torch.float32)

        if self.return_meta:
            meta = dict(self._meta_by_path.get(path, {}))
            meta.setdefault("path", path)
            return x, y, meta
        return x, y
