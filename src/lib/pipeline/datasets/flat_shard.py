"""Dataset adapter for BuildAI-like flat shard directories."""

from __future__ import annotations

from pathlib import Path

from lib.pipeline.datasets.base import (
    AdapterValidationResult,
    BaseDatasetAdapter,
    register_dataset_adapter,
)
from lib.pipeline.io.video_index import collect_videos_from_factory


@register_dataset_adapter
class FlatShardDatasetAdapter(BaseDatasetAdapter):
    name = "flat_shard"

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
    ):
        shard_dir = Path(
            adapter_cfg.get("shard_dir")
            or paths_cfg.get("shard_dir")
            or paths_cfg.get("shard_root", "")
        )
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"shard_dir not found: {shard_dir}")

        seq_folder_root = Path(adapter_cfg.get("seq_folder_root") or (shard_dir / "outputs"))
        descriptors = collect_videos_from_factory(str(shard_dir.resolve()))
        for descriptor in descriptors:
            descriptor.seq_folder = str((seq_folder_root / descriptor.clip_id).resolve())
        return descriptors

    def validate_source(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared=None,
        manifest_records=None,
    ) -> AdapterValidationResult:
        shard_dir = Path(
            adapter_cfg.get("shard_dir")
            or paths_cfg.get("shard_dir")
            or paths_cfg.get("shard_root", "")
        )
        if not shard_dir.is_dir():
            return AdapterValidationResult(
                ok=False,
                summary={
                    "adapter": self.name,
                    "error": f"shard_dir not found: {shard_dir}",
                },
            )

        tar_files = sorted(path.name for path in shard_dir.iterdir() if path.is_file() and path.suffix == ".tar")
        outputs_dir = Path(adapter_cfg.get("seq_folder_root") or (shard_dir / "outputs"))
        return AdapterValidationResult(
            ok=bool(tar_files),
            summary={
                "adapter": self.name,
                "shard_dir": str(shard_dir.resolve()),
                "seq_folder_root": str(outputs_dir.resolve()),
                "shard_count": len(tar_files),
                "has_outputs_dir": outputs_dir.is_dir(),
            },
        )
