"""Shared helpers and MANO face constants for the HaWoR pipeline stages."""

import numpy as np

from lib.pipeline.proc.logging_setup import QUIET_MODE, vprint  # noqa: F401

MANO_FACE_EXTRA = np.array(
    [
        [92, 38, 234],
        [234, 38, 239],
        [38, 122, 239],
        [239, 122, 279],
        [122, 118, 279],
        [279, 118, 215],
        [118, 117, 215],
        [215, 117, 214],
        [117, 119, 214],
        [214, 119, 121],
        [119, 120, 121],
        [121, 120, 78],
        [120, 108, 78],
        [78, 108, 79],
    ],
    dtype=np.int32,
)
