import argparse
from pathlib import Path


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {dst}")


def _insert_once(text: str, marker: str, insert_text: str) -> str:
    if insert_text.strip() in text:
        return text
    if marker not in text:
        raise RuntimeError(f"Marker not found: {marker!r}")
    return text.replace(marker, f"{insert_text}\n{marker}", 1)


def _patch_init(openstl_root: Path) -> None:
    path = openstl_root / "openstl" / "datasets" / "__init__.py"
    text = path.read_text(encoding="utf-8")
    import_line = "from .fus_npz_dataset import FUSNPZForecastDataset"
    if import_line not in text:
        text = _insert_once(text, "from .dataloader import load_data", import_line)
    if "'FUSNPZForecastDataset'" not in text and "__all__" in text:
        text = text.replace(
            "'SEVIRDataset'",
            "'SEVIRDataset', 'FUSNPZForecastDataset'",
            1,
        )
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


def _patch_dataloader(openstl_root: Path) -> None:
    path = openstl_root / "openstl" / "datasets" / "dataloader.py"
    text = path.read_text(encoding="utf-8")
    if "dataloader_fus_npz" in text:
        print(f"already patched {path}")
        return
    marker = (
        "    else:\n"
        "        raise ValueError(f'Dataname {dataname} is unsupported')"
    )
    insert_block = (
        "    elif dataname in ['fus', 'fus_npz']:\n"
        "        from .dataloader_fus_npz import load_data\n"
        "        cfg_dataloader['manifest_path'] = kwargs.get('manifest_path', None)\n"
        "        cfg_dataloader['frames_key'] = kwargs.get('frames_key', 'frames')\n"
        "        cfg_dataloader['stride'] = kwargs.get('stride', 1)\n"
        "        cfg_dataloader['seed'] = kwargs.get('seed', 42)\n"
        "        cfg_dataloader['lru_cache_size'] = kwargs.get('lru_cache_size', 8)\n"
        "        return load_data(\n"
        "            batch_size, val_batch_size,\n"
        "            data_root, num_workers, **cfg_dataloader\n"
        "        )\n"
    )
    text = _insert_once(text, marker, insert_block)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


def _patch_dataset_constant(openstl_root: Path) -> None:
    path = openstl_root / "openstl" / "datasets" / "dataset_constant.py"
    text = path.read_text(encoding="utf-8")
    if "'fus_npz'" in text:
        print(f"already patched {path}")
        return
    anchor = "\n}"
    block = (
        "    'fus_npz': {\n"
        "        'in_shape': [20, 1, 128, 128],\n"
        "        'pre_seq_length': 20,\n"
        "        'aft_seq_length': 20,\n"
        "        'total_length': 40,\n"
        "        'frames_key': 'frames',\n"
        "        'stride': 1,\n"
        "        'seed': 42,\n"
        "        'metrics': ['mse', 'mae', 'ssim', 'psnr'],\n"
        "    },\n"
    )
    if anchor not in text:
        raise RuntimeError(f"Could not patch {path}; closing brace not found.")
    text = text.replace(anchor, f"{block}{anchor}", 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


def _patch_parser(openstl_root: Path) -> None:
    path = openstl_root / "openstl" / "utils" / "parser.py"
    text = path.read_text(encoding="utf-8")
    if "'fus_npz'" in text:
        print(f"already patched {path}")
        return
    if "'sevir_vil'" not in text:
        raise RuntimeError(
            f"Could not patch {path}; expected dataname choices not found."
        )
    text = text.replace("'sevir_vil'],", "'sevir_vil', 'fus_npz'],", 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply fUS NPZ patch to an OpenSTL checkout."
    )
    parser.add_argument(
        "--openstl-root",
        required=True,
        help="Path to OpenSTL repo root",
    )
    args = parser.parse_args()

    openstl_root = Path(args.openstl_root).expanduser().resolve(strict=False)
    patch_root = Path(__file__).resolve().parent

    _copy_file(
        patch_root / "openstl" / "datasets" / "fus_npz_dataset.py",
        openstl_root / "openstl" / "datasets" / "fus_npz_dataset.py",
    )
    _copy_file(
        patch_root / "openstl" / "datasets" / "dataloader_fus_npz.py",
        openstl_root / "openstl" / "datasets" / "dataloader_fus_npz.py",
    )
    _copy_file(
        patch_root / "configs" / "fus" / "SimVP.py",
        openstl_root / "configs" / "fus" / "SimVP.py",
    )

    _patch_init(openstl_root)
    _patch_dataloader(openstl_root)
    _patch_dataset_constant(openstl_root)
    _patch_parser(openstl_root)

    print("OpenSTL fUS NPZ patch applied.")


if __name__ == "__main__":
    main()
