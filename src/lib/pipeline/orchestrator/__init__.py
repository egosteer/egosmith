"""Dataset pipeline orchestration helpers."""

__all__ = ["main", "run_pipeline"]


def __getattr__(name):
    if name in __all__:
        from . import pipeline

        value = getattr(pipeline, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
