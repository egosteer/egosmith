"""Dataset-level distribution summaries (percentiles, IQR fences) over per-clip metrics."""

from __future__ import annotations

import numpy as np


def summarize_metric_distribution(values) -> dict | None:
    finite = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return None
    return {
        "count": int(finite.size),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite.max()),
    }


def summarize_iqr_distribution(values, multiplier: float) -> dict | None:
    finite = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return None
    q1 = float(np.percentile(finite, 25))
    q3 = float(np.percentile(finite, 75))
    iqr = float(q3 - q1)
    return {
        "count": int(finite.size),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "lower_bound": float(q1 - multiplier * iqr),
        "upper_bound": float(q3 + multiplier * iqr),
    }
