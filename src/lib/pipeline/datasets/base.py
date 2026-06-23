"""Dataset adapter protocol for source-specific pipeline integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from lib.pipeline.datasets.descriptors import ClipDescriptor


@dataclass(frozen=True)
class DatasetAdapterContext:
    project_root: Path
    run_dir: Path
    manifest_path: Path
    shard_dirs_list_path: Path
    summary_path: Path


@dataclass
class AdapterPrepareResult:
    payload: dict = field(default_factory=dict)


@dataclass
class AdapterValidationResult:
    ok: bool = True
    summary: dict = field(default_factory=dict)


class BaseDatasetAdapter:
    name = "base"

    def prepare(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        runtimes_cfg: dict,
        context: DatasetAdapterContext,
        run_logged: Optional[Callable[..., None]] = None,
    ) -> AdapterPrepareResult:
        return AdapterPrepareResult()

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context: Optional[DatasetAdapterContext] = None,
        prepared: Optional[AdapterPrepareResult] = None,
    ) -> Iterable[ClipDescriptor]:
        raise NotImplementedError

    def resolve_annotation_context(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context: DatasetAdapterContext,
        prepared: Optional[AdapterPrepareResult] = None,
    ) -> dict:
        return {}

    def validate_source(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context: Optional[DatasetAdapterContext] = None,
        prepared: Optional[AdapterPrepareResult] = None,
        manifest_records=None,
    ) -> AdapterValidationResult:
        return AdapterValidationResult()


_ADAPTER_REGISTRY: dict[str, type[BaseDatasetAdapter]] = {}


def register_dataset_adapter(adapter_cls: type[BaseDatasetAdapter]) -> type[BaseDatasetAdapter]:
    _ADAPTER_REGISTRY[adapter_cls.name] = adapter_cls
    return adapter_cls


def get_dataset_adapter(name: str) -> BaseDatasetAdapter:
    try:
        adapter_cls = _ADAPTER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset adapter: {name}. Available: {sorted(_ADAPTER_REGISTRY)}") from exc
    return adapter_cls()
