"""BuildAI dataset adapter."""

from __future__ import annotations

from pathlib import Path

from lib.pipeline.clips.clip_manifest import discover_shard_dirs, remap_descriptor_seq_folders
from lib.pipeline.datasets.base import (
    AdapterPrepareResult,
    AdapterValidationResult,
    BaseDatasetAdapter,
    register_dataset_adapter,
)
from lib.pipeline.io.video_index import collect_videos_from_factories


def _buildai_group_names(start_factory_id: int, end_factory_id: int) -> list[str]:
    return [f"factory{factory_id:03d}" for factory_id in range(start_factory_id, end_factory_id + 1)]


@register_dataset_adapter
class BuildAIDatasetAdapter(BaseDatasetAdapter):
    name = "buildai"

    def prepare(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        runtimes_cfg: dict,
        context,
        run_logged=None,
    ) -> AdapterPrepareResult:
        if run_logged is None:
            return AdapterPrepareResult()

        start_factory_id = int(dataset_cfg["start_factory_id"])
        end_factory_id = int(dataset_cfg["end_factory_id"])
        buildai_repo_root = Path(paths_cfg.get("buildai_repo_root", "/root/buildai_processing"))
        buildai_config = paths_cfg.get("buildai_config")
        preprocess_cmd = [
            runtimes_cfg.get("buildai_shell", "/bin/bash"),
            str(buildai_repo_root / "run_buildai_pipeline.sh"),
            "--config",
            str(buildai_config),
            "--start-factory-id",
            str(start_factory_id),
            "--end-factory-id",
            str(end_factory_id),
            "--stages",
            adapter_cfg.get("stages", "1,2,3"),
        ]
        if adapter_cfg.get("setup_decord"):
            preprocess_cmd.append("--setup-decord")
        if adapter_cfg.get("clean_stage3_output"):
            preprocess_cmd.append("--clean-stage3-output")
        run_logged("preprocess", preprocess_cmd, cwd=buildai_repo_root)
        return AdapterPrepareResult(
            payload={
                "include_dirs": _buildai_group_names(start_factory_id, end_factory_id),
            }
        )

    def build_descriptors(
        self,
        *,
        dataset_cfg: dict,
        adapter_cfg: dict,
        paths_cfg: dict,
        context=None,
        prepared: AdapterPrepareResult | None = None,
    ):
        shard_root = Path(paths_cfg["shard_root"])
        include_dirs = None
        if prepared is not None:
            include_dirs = prepared.payload.get("include_dirs")
        if include_dirs is None:
            include_dirs = _buildai_group_names(
                int(dataset_cfg["start_factory_id"]),
                int(dataset_cfg["end_factory_id"]),
            )
        shard_dirs = discover_shard_dirs(shard_root, include_dirs=include_dirs)
        descriptors = collect_videos_from_factories(shard_dirs)

        seq_folder_root = (
            adapter_cfg.get("seq_folder_root")
            or paths_cfg.get("seq_folder_root")
            or paths_cfg.get("stage_output_root")
        )
        if seq_folder_root:
            remap_descriptor_seq_folders(descriptors, seq_folder_root)
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
        start_factory_id = int(dataset_cfg["start_factory_id"])
        end_factory_id = int(dataset_cfg["end_factory_id"])
        expected_dirs = _buildai_group_names(start_factory_id, end_factory_id)
        shard_root = Path(paths_cfg["shard_root"])
        present_dirs = [name for name in expected_dirs if (shard_root / name).is_dir()]
        missing_dirs = [name for name in expected_dirs if not (shard_root / name).is_dir()]
        seq_folder_root = (
            adapter_cfg.get("seq_folder_root")
            or paths_cfg.get("seq_folder_root")
            or paths_cfg.get("stage_output_root")
        )
        return AdapterValidationResult(
            ok=len(missing_dirs) == 0,
            summary={
                "adapter": self.name,
                "expected_group_count": len(expected_dirs),
                "present_group_count": len(present_dirs),
                "missing_groups": missing_dirs,
                "seq_folder_root": seq_folder_root,
            },
        )
