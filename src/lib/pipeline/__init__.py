__all__ = [
    "PipelineVideoTask",
    "STAGES",
    "StageExecutionConfig",
    "get_seq_folder",
    "get_stage_done_marker",
    "get_track_range",
    "get_tracks_dir",
    "is_stage_complete",
    "run_pipeline_stage",
    "validate_stage_output",
    "validate_stage_output_fast",
]


def __getattr__(name):
    if name in __all__:
        from . import stage_api

        value = getattr(stage_api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
