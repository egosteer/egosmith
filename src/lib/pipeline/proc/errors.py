"""Pipeline-specific exception types."""


class CorruptStageDataError(RuntimeError):
    """Raised when a video's stage inputs or caches are corrupt and the video should be skipped."""

