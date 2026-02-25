import os

from openstl.datasets.fus_npz_dataset import FUSNPZForecastDataset
from openstl.datasets.utils import create_loader


def load_data(
    batch_size,
    val_batch_size,
    data_root,
    num_workers=4,
    pre_seq_length=20,
    aft_seq_length=20,
    in_shape=None,
    distributed=False,
    use_augment=False,
    use_prefetcher=False,
    drop_last=False,
    manifest_path=None,
    frames_key="frames",
    stride=1,
    seed=42,
    lru_cache_size=8,
):
    del use_augment  # Not used for NPZ manifest loading.
    if manifest_path is None:
        manifest_path = os.path.join(data_root, "fus", "splits_multi.json")

    train_set = FUSNPZForecastDataset(
        manifest_path=manifest_path,
        split="train",
        frames_key=frames_key,
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        stride=stride,
        random_window=True,
        seed=seed,
        lru_cache_size=lru_cache_size,
    )
    val_set = FUSNPZForecastDataset(
        manifest_path=manifest_path,
        split="val",
        frames_key=frames_key,
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        stride=stride,
        random_window=False,
        seed=seed,
        lru_cache_size=lru_cache_size,
    )
    test_set = FUSNPZForecastDataset(
        manifest_path=manifest_path,
        split="test",
        frames_key=frames_key,
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        stride=stride,
        random_window=False,
        seed=seed,
        lru_cache_size=lru_cache_size,
    )

    in_channels = 1
    if isinstance(in_shape, (list, tuple)) and len(in_shape) >= 2:
        in_channels = int(in_shape[1])

    dataloader_train = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
        input_channels=in_channels,
    )
    dataloader_vali = create_loader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
        input_channels=in_channels,
    )
    dataloader_test = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True,
        distributed=distributed,
        use_prefetcher=use_prefetcher,
        input_channels=in_channels,
    )

    return dataloader_train, dataloader_vali, dataloader_test
