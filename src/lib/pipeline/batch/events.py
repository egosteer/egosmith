"""Structured JSONL event logging for a batch run (BatchEventLogger)."""

import json
from datetime import datetime, timezone
from pathlib import Path


class BatchEventLogger:
    def __init__(self, events_file: Path):
        self.events_file = events_file

    def emit(self, event: str, **kwargs):
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **kwargs,
        }
        with open(self.events_file, "a") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
