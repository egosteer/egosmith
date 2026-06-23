"""Dataset adapters and canonical descriptor types."""

from importlib import import_module

from lib.pipeline.datasets.base import (
    AdapterPrepareResult,
    AdapterValidationResult,
    BaseDatasetAdapter,
    DatasetAdapterContext,
    get_dataset_adapter as _get_registered_dataset_adapter,
    register_dataset_adapter,
)
from lib.pipeline.datasets.descriptors import ClipDescriptor, STORAGE_IMAGE_SEQUENCE, STORAGE_TAR_SHARD

_BUILTIN_ADAPTER_MODULES = {
    "buildai": "lib.pipeline.datasets.buildai",
    "flat_shard": "lib.pipeline.datasets.flat_shard",
    "fpha_tar": "lib.pipeline.datasets.fpha_tar",
    "image_sequence": "lib.pipeline.datasets.image_sequence",
    "hot3d_wds": "lib.pipeline.datasets.hot3d_wds",
    "legacy_buildai": "lib.pipeline.datasets.legacy_buildai",
    "single_video": "lib.pipeline.datasets.single_video",
    "video_folder": "lib.pipeline.datasets.video_folder",
}


def get_dataset_adapter(name: str):
    module_name = _BUILTIN_ADAPTER_MODULES.get(name)
    if module_name is not None:
        import_module(module_name)
    return _get_registered_dataset_adapter(name)


__all__ = [
    "AdapterPrepareResult",
    "AdapterValidationResult",
    "BaseDatasetAdapter",
    "ClipDescriptor",
    "DatasetAdapterContext",
    "STORAGE_IMAGE_SEQUENCE",
    "STORAGE_TAR_SHARD",
    "get_dataset_adapter",
    "register_dataset_adapter",
]
