import os
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_ENV_VAR = "HAWOR_CLIP_CONFIG"
LEGACY_CONFIG_ENV_VAR = "BUILDAI_PIPELINE_CONFIG"
DEFAULT_CONFIG_NAME = "heuristic_clip_config.yaml"


@lru_cache(maxsize=None)
def get_pipeline_config(config_path=None):
    raw_path = config_path or os.environ.get(CONFIG_ENV_VAR) or os.environ.get(LEGACY_CONFIG_ENV_VAR)
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
    else:
        path = Path(__file__).resolve().with_name(DEFAULT_CONFIG_NAME)

    with open(path, "r") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def get_pipeline_config_path(config_path=None):
    raw_path = config_path or os.environ.get(CONFIG_ENV_VAR) or os.environ.get(LEGACY_CONFIG_ENV_VAR)
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return Path(__file__).resolve().with_name(DEFAULT_CONFIG_NAME)
