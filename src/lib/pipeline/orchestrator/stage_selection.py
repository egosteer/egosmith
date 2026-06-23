"""Stage selection and compatibility handling for the orchestrator."""

from __future__ import annotations

from .constants import INTERNAL_STAGE_ORDER, LEGACY_STAGE_NAMES, OFFICIAL_STAGE_ORDER, STAGE_ALIAS_MAP


def selected_stages(raw: str) -> dict:
    requested = [stage.strip() for stage in raw.split(",") if stage.strip()]
    valid = set(OFFICIAL_STAGE_ORDER) | set(INTERNAL_STAGE_ORDER)
    invalid = [stage for stage in requested if stage not in valid]
    if invalid:
        raise ValueError(
            f"Unknown stages: {invalid}. Valid official stages: {OFFICIAL_STAGE_ORDER}. "
            f"Legacy compatibility stages: {sorted(LEGACY_STAGE_NAMES)}"
        )

    expanded = set()
    requested_public = []
    deprecated = []
    for stage in requested:
        if stage in STAGE_ALIAS_MAP:
            canonical = stage
            internal_names = STAGE_ALIAS_MAP[stage]
        else:
            if stage in LEGACY_STAGE_NAMES:
                deprecated.append(stage)
            if stage in ("preprocess", "manifest"):
                canonical = "prepare"
            elif stage in ("detect_motion", "slam", "infiller"):
                canonical = "infer"
            else:
                canonical = stage
            internal_names = [stage]

        if canonical not in requested_public:
            requested_public.append(canonical)
        expanded.update(internal_names)

    return {
        "requested_tokens": requested,
        "requested_public": requested_public,
        "internal": [stage for stage in INTERNAL_STAGE_ORDER if stage in expanded],
        "deprecated": deprecated,
    }
