"""Shared constants for the dataset pipeline orchestrator."""

OFFICIAL_STAGE_ORDER = [
    "prepare",
    "annotate",
    "infer",
    "filter",
    "build",
    "validate",
]

INTERNAL_STAGE_ORDER = [
    "preprocess",
    "manifest",
    "annotate",
    "detect_motion",
    "slam",
    "native_depth",
    "infiller",
    "filter",
    "build",
    "validate",
]

STAGE_ALIAS_MAP = {
    "prepare": ["preprocess", "manifest"],
    "infer": ["detect_motion", "slam", "infiller"],
}

LEGACY_STAGE_NAMES = {"preprocess", "manifest", "detect_motion", "slam", "infiller"}

BATCH_INFER_NEGATIVE_BOOL_FLAGS = {
    "resume",
    "detect_half_precision",
    "depth_predict_all_frames",
    "any4d_use_amp",
}

MULTIHOST_DISALLOWED_INFER_KEYS = {
    "descriptor_manifest",
    "video_list",
    "video_dir",
    "run_dir",
    "start",
    "end",
    "stages",
}
