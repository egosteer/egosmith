"""Validation helpers for pipeline configs and multihost orchestration."""

from __future__ import annotations

import argparse

from lib.pipeline.batch.cli import build_batch_infer_parser

from .constants import MULTIHOST_DISALLOWED_INFER_KEYS
from .helpers import parser_supported_option_dests, validate_cli_mapping_keys


def validate_pipeline_cli_alignment(*, stages: list[str], infer_cfg: dict, build_cfg: dict, filter_cfg: dict, validation_cfg: dict) -> None:
    errors = []

    if any(stage in stages for stage in ("detect_motion", "slam", "infiller")):
        infer_supported = parser_supported_option_dests(build_batch_infer_parser())
        infer_reserved = {"descriptor_manifest", "video_list", "video_dir", "run_dir"}
        errors.extend(
            validate_cli_mapping_keys(
                label=f"infer.{section}",
                mapping=infer_cfg.get(section),
                supported_keys=infer_supported,
                reserved_keys=infer_reserved,
            )
            for section in ("common", "detect_motion", "slam", "infiller")
        )
    if "native_depth" in stages or bool((infer_cfg.get("native_depth") or {}).get("enabled")):
        from scripts.build.run_hot3d_native_depth import build_parser as get_native_depth_parser

        native_depth_supported = parser_supported_option_dests(get_native_depth_parser())
        native_depth_reserved = {"descriptor_manifest", "run_dir", "report_out"}
        errors.extend(
            validate_cli_mapping_keys(
                label="infer.native_depth",
                mapping={
                    key: value
                    for key, value in (infer_cfg.get("native_depth") or {}).items()
                    if key != "enabled"
                },
                supported_keys=native_depth_supported,
                reserved_keys=native_depth_reserved,
            )
        )

    if "build" in stages:
        from scripts.build.build_vla_from_manifest import get_parser as get_build_parser

        build_supported = parser_supported_option_dests(get_build_parser())
        build_reserved = {"descriptor_manifest", "output_dir", "annotation_root"}
        errors.extend(
            validate_cli_mapping_keys(
                label="build",
                mapping=build_cfg,
                supported_keys=build_supported,
                reserved_keys=build_reserved,
            )
        )

    if "filter" in stages:
        from scripts.build.filter_manifest_by_quality import build_parser as get_filter_parser

        filter_supported = parser_supported_option_dests(get_filter_parser())
        filter_reserved = {"input_manifest", "output_manifest", "report_out"}
        errors.extend(
            validate_cli_mapping_keys(
                label="filter",
                mapping=filter_cfg,
                supported_keys=filter_supported,
                reserved_keys=filter_reserved,
            )
        )

    if "validate" in stages:
        from scripts.inspection.validate_pipeline_run import get_parser as get_validate_parser

        validate_supported = parser_supported_option_dests(get_validate_parser())
        validate_reserved = {"descriptor_manifest", "dataset_dir", "annotation_root", "annotation_suffix"}
        errors.extend(
            validate_cli_mapping_keys(
                label="validation",
                mapping=validation_cfg,
                supported_keys=validate_supported,
                reserved_keys=validate_reserved,
            )
        )

    flat_errors = [entry for group in errors for entry in (group if isinstance(group, list) else [group]) if entry]
    if flat_errors:
        raise ValueError("Pipeline config contains unsupported child CLI keys:\n- " + "\n- ".join(flat_errors))


def validate_multihost_infer_alignment(infer_cfg: dict) -> None:
    errors = []
    for section in ("common", "detect_motion", "slam", "infiller"):
        mapping = infer_cfg.get(section) or {}
        invalid = sorted(key for key in mapping if key in MULTIHOST_DISALLOWED_INFER_KEYS)
        if invalid:
            errors.append(f"infer.{section}: unsupported under multihost {invalid}")
    if errors:
        raise ValueError("infer.multihost uses reserved batch_infer keys:\n- " + "\n- ".join(errors))


def infer_stage_worker_count_per_gpu(*, pipeline_stage: str, infer_cfg: dict) -> int:
    common_cfg = infer_cfg.get("common") or {}
    stage_cfg = infer_cfg.get(pipeline_stage) or {}
    internal_stage_map = {
        "detect_motion": ("detect_track", "motion"),
        "slam": ("slam",),
        "infiller": ("infiller",),
    }
    internal_stages = internal_stage_map[pipeline_stage]
    default_workers = int(stage_cfg.get("workers_per_gpu", common_cfg.get("workers_per_gpu", 1)))
    resolved = []
    for internal_stage in internal_stages:
        key = f"{internal_stage}_workers_per_gpu"
        resolved.append(int(stage_cfg.get(key, common_cfg.get(key, default_workers))))
    return max(resolved)
