import json
from pathlib import Path
from typing import Any


def normalize_manifest_path(path: str | Path) -> str:
    """Normalize a path for JSON manifests with Windows-safe separators."""
    return Path(path).expanduser().resolve(strict=False).as_posix()


def _is_path_key(key: str) -> bool:
    return key == "path" or key.endswith("_path")


def normalize_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    out = dict(manifest)

    for split_key in ("train", "val", "test"):
        values = out.get(split_key)
        if isinstance(values, list):
            out[split_key] = [normalize_manifest_path(p) for p in values]

    meta = out.get("meta")
    if isinstance(meta, dict):
        norm_meta: dict[str, Any] = {}
        for path_key, info in meta.items():
            norm_key = normalize_manifest_path(path_key)
            if not isinstance(info, dict):
                norm_meta[norm_key] = info
                continue
            norm_info: dict[str, Any] = {}
            for k, v in info.items():
                if isinstance(v, (str, Path)) and _is_path_key(str(k)):
                    norm_info[str(k)] = normalize_manifest_path(v)
                else:
                    norm_info[str(k)] = v
            norm_meta[norm_key] = norm_info
        out["meta"] = norm_meta

    acquisitions = out.get("acquisitions")
    if isinstance(acquisitions, list):
        norm_acquisitions: list[dict[str, Any]] = []
        for item in acquisitions:
            if not isinstance(item, dict):
                continue
            norm_item: dict[str, Any] = {}
            for k, v in item.items():
                if isinstance(v, (str, Path)) and _is_path_key(str(k)):
                    norm_item[str(k)] = normalize_manifest_path(v)
                else:
                    norm_item[str(k)] = v
            norm_acquisitions.append(norm_item)
        out["acquisitions"] = norm_acquisitions

    if "cache_dir" in out and isinstance(out["cache_dir"], (str, Path)):
        out["cache_dir"] = normalize_manifest_path(out["cache_dir"])
    if "subject_to_cache_dir" in out and isinstance(out["subject_to_cache_dir"], dict):
        out["subject_to_cache_dir"] = {
            str(k): normalize_manifest_path(v)
            for k, v in out["subject_to_cache_dir"].items()
        }

    return out


def _iter_manifest_paths(manifest: dict[str, Any]):
    for split_key in ("train", "val", "test"):
        values = manifest.get(split_key, [])
        if isinstance(values, list):
            for p in values:
                yield str(p), split_key

    meta = manifest.get("meta", {})
    if isinstance(meta, dict):
        for path_key, info in meta.items():
            yield str(path_key), "meta"
            if isinstance(info, dict):
                for k, v in info.items():
                    if isinstance(v, str) and _is_path_key(str(k)):
                        yield v, f"meta.{k}"

    acquisitions = manifest.get("acquisitions", [])
    if isinstance(acquisitions, list):
        for idx, item in enumerate(acquisitions):
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if isinstance(v, str) and _is_path_key(str(k)):
                    yield v, f"acquisitions[{idx}].{k}"


def assert_manifest_paths_exist(manifest_path: str | Path) -> int:
    manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    checked = 0
    missing: list[str] = []
    for path_str, context in _iter_manifest_paths(manifest):
        checked += 1
        if not Path(path_str).exists():
            missing.append(f"{context}: {path_str}")
    if missing:
        preview = "\n".join(missing[:5])
        raise AssertionError(
            "Manifest references missing paths "
            f"({len(missing)} / {checked} checked):\n{preview}"
        )
    return checked
