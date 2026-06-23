__all__ = ["BatchRunConfig", "BatchScheduler", "STAGE_ALIASES"]


def __getattr__(name):
    if name in {"BatchRunConfig", "STAGE_ALIASES"}:
        from lib.pipeline.batch import config

        value = getattr(config, name)
        globals()[name] = value
        return value
    if name == "BatchScheduler":
        from lib.pipeline.batch import scheduler

        value = scheduler.BatchScheduler
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
