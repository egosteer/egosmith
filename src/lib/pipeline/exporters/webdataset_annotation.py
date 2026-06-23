"""Annotation helpers for WebDataset export and rewrite flows."""

import json
import os
from multiprocessing import get_context

from tqdm import tqdm

from lib.pipeline.clips.annotation_protocol import strip_leading_instruction_numbering

ANNOTATION_LEVEL_KEYS = ("level1", "level2", "level3", "level4", "level5")
DEFAULT_ANNOTATION_SUFFIX = "_qwen-annotation.json"


def get_episode_annotation_path(ep, annotation_suffix=DEFAULT_ANNOTATION_SUFFIX):
    """Return the qwen-annotation JSON path for an episode."""
    parent_dir = os.path.dirname(ep["crop_dir"])
    return os.path.join(parent_dir, f"{ep['episode_id']}{annotation_suffix}")


def normalize_instruction(global_analysis):
    """Normalize level1..level5 annotation text into an instruction list."""
    if not isinstance(global_analysis, dict):
        return []

    instruction = []
    for level_key in ANNOTATION_LEVEL_KEYS:
        value = global_analysis.get(level_key)
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        value = strip_leading_instruction_numbering(value)
        if value:
            instruction.append(value)
    return instruction


def load_episode_instruction(ep, annotation_suffix=DEFAULT_ANNOTATION_SUFFIX):
    """Load and validate one episode's qwen annotation."""
    annotation_path = get_episode_annotation_path(ep, annotation_suffix=annotation_suffix)
    if not os.path.exists(annotation_path):
        return None, "missing_annotation", annotation_path

    try:
        with open(annotation_path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, "invalid_json", annotation_path

    status = str(payload.get("status", "")).strip()
    if status != "Valid":
        return None, "invalid_status", annotation_path

    instruction = normalize_instruction(payload.get("global_analysis"))
    if not instruction:
        return None, "empty_instruction", annotation_path

    return instruction, None, annotation_path


def _load_episode_instruction_worker(task):
    ep, annotation_suffix = task
    instruction, error_code, annotation_path = load_episode_instruction(ep, annotation_suffix=annotation_suffix)
    return ep, instruction, error_code, annotation_path


def attach_or_filter_episode_instructions(
    episodes,
    annotation_suffix=DEFAULT_ANNOTATION_SUFFIX,
    allow_missing_annotation=False,
    workers=1,
):
    """Attach instruction to episodes or drop invalid entries."""
    kept = []
    stats = {
        "kept": 0,
        "filtered": 0,
        "missing_annotation": 0,
        "invalid_json": 0,
        "invalid_status": 0,
        "empty_instruction": 0,
    }

    if workers <= 1:
        iterator = (
            (ep, *load_episode_instruction(ep, annotation_suffix=annotation_suffix))
            for ep in episodes
        )
    else:
        tasks = ((ep, annotation_suffix) for ep in episodes)
        mp_context = get_context()
        pool = mp_context.Pool(workers)
        iterator = pool.imap(_load_episode_instruction_worker, tasks, chunksize=64)

    try:
        for item in tqdm(iterator, total=len(episodes), desc="Episode annotations"):
            ep, instruction, error_code, annotation_path = item
            ep_copy = dict(ep)
            ep_copy["annotation_path"] = annotation_path

            if instruction is None:
                stats[error_code] = stats.get(error_code, 0) + 1
                if allow_missing_annotation:
                    ep_copy["instruction"] = []
                    kept.append(ep_copy)
                    stats["kept"] += 1
                else:
                    stats["filtered"] += 1
                continue

            ep_copy["instruction"] = instruction
            kept.append(ep_copy)
            stats["kept"] += 1
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    return kept, stats
