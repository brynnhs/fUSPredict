from __future__ import annotations

from pathlib import Path
from typing import Any

SUPPORTED_MODES = ("raw", "mean_divide", "zscore")
SUPPORTED_CONDITIONS = ("unfiltered", "filtered")
ALL_TOKEN = "all"

RAW_STAGE_BY_CONDITION = {
    "unfiltered": "reoriented_resized",
    "filtered": "filtered",
}


def _normalize_choice_list(
    value: Any,
    *,
    supported: tuple[str, ...],
    field_name: str,
    default: str | list[str],
) -> list[str]:
    src = default if value is None else value

    if isinstance(src, str):
        items = [src]
    elif isinstance(src, (list, tuple)):
        items = list(src)
    else:
        raise TypeError(f"{field_name} must be a string, list, tuple, or null; got {type(src)!r}")

    normalized: list[str] = []
    for raw_item in items:
        item = str(raw_item).strip().lower()
        if item == ALL_TOKEN:
            return list(supported)
        if item not in supported:
            allowed = ", ".join(supported)
            raise ValueError(f"Unsupported {field_name}={raw_item!r}; expected one of: {allowed}, {ALL_TOKEN}")
        if item not in normalized:
            normalized.append(item)

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_default(value: Any, *, field_name: str, selected: list[str]) -> str:
    default = str(value if value is not None else selected[0]).strip().lower()
    if default not in selected:
        raise ValueError(f"{field_name}={value!r} must be one of the selected values: {selected}")
    return default


def get_combo_id(mode: str, condition: str) -> str:
    mode = str(mode).strip().lower()
    condition = str(condition).strip().lower()
    if mode == "raw":
        return f"raw_{condition}"
    return f"{condition}_{mode}"


def get_npz_pattern(mode: str, condition: str) -> str:
    mode = str(mode).strip().lower()
    condition = str(condition).strip().lower()

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode={mode!r}; expected one of {SUPPORTED_MODES}")
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"Unsupported condition={condition!r}; expected one of {SUPPORTED_CONDITIONS}")

    if mode == "raw":
        suffix = RAW_STAGE_BY_CONDITION[condition]
        return f"baseline_*_{suffix}.npz"
    return f"baseline_*_{condition}_standardized_{mode}.npz"


def get_cache_subdir(mode: str, condition: str) -> str:
    mode = str(mode).strip().lower()
    condition = str(condition).strip().lower()

    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode={mode!r}; expected one of {SUPPORTED_MODES}")
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"Unsupported condition={condition!r}; expected one of {SUPPORTED_CONDITIONS}")

    if mode == "raw":
        if condition == "filtered":
            return "baseline_only_filtered"
        return "baseline_only_reoriented_resized"
    return "baseline_only_standardized"


def get_cache_dir(preprocessing_root: str | Path, subject: str, mode: str, condition: str) -> Path:
    return Path(preprocessing_root) / str(subject) / get_cache_subdir(mode, condition)


def build_combo_spec(mode: str, condition: str) -> dict[str, str]:
    mode = str(mode).strip().lower()
    condition = str(condition).strip().lower()
    return {
        "combo_id": get_combo_id(mode, condition),
        "data_mode": mode,
        "processing_condition": condition,
        "pattern": get_npz_pattern(mode, condition),
        "output_suffix": get_combo_id(mode, condition),
    }


def resolve_data_selection(config: dict[str, Any]) -> dict[str, Any]:
    selection_cfg = config.get("data_selection", {})
    if not isinstance(selection_cfg, dict):
        raise TypeError("config['data_selection'] must be a mapping")

    modes = _normalize_choice_list(
        selection_cfg.get("modes"),
        supported=SUPPORTED_MODES,
        field_name="data_selection.modes",
        default="zscore",
    )
    conditions = _normalize_choice_list(
        selection_cfg.get("conditions"),
        supported=SUPPORTED_CONDITIONS,
        field_name="data_selection.conditions",
        default=ALL_TOKEN,
    )
    default_mode = _normalize_default(
        selection_cfg.get("default_mode"),
        field_name="data_selection.default_mode",
        selected=modes,
    )
    default_condition = _normalize_default(
        selection_cfg.get("default_condition"),
        field_name="data_selection.default_condition",
        selected=conditions,
    )

    combos = [build_combo_spec(mode, condition) for condition in conditions for mode in modes]
    combo_lookup = {combo["combo_id"]: combo for combo in combos}

    return {
        "modes": modes,
        "conditions": conditions,
        "default_mode": default_mode,
        "default_condition": default_condition,
        "combos": combos,
        "combo_lookup": combo_lookup,
    }
