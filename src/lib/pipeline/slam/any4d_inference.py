"""Any4D inference entry points used by the EgoSmith depth stage.

These three functions are taken **verbatim** from upstream Any4D's
``scripts/demo_inference.py`` (https://github.com/Any-4D/Any4D, Apache-2.0):
``init_hydra_config``, ``init_inference_model`` and ``sample_inference``.

They are carried here as first-party code so EgoSmith does not depend on Any4D's
``scripts/`` directory (which is not part of the shipped Any4D core). Only the
``any4d`` package itself (installed via ``pip install -e thirdparty/Any4D``) and
its Hydra configs are required at runtime. No logic is changed; the bodies match
upstream byte-for-byte.
"""

import os

import hydra
import torch

from any4d.models import init_model
from any4d.utils.inference import loss_of_one_batch_multi_view


def init_hydra_config(config_path, overrides=None):
    "Initialize Hydra config"
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path).split(".")[0]
    relative_path = os.path.relpath(config_dir, os.path.dirname(__file__))
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=relative_path)
    if overrides is not None:
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    else:
        cfg = hydra.compose(config_name=config_name)

    return cfg


def init_inference_model(config, ckpt_path, device):
    "Initialize the model for inference"
    # Load the model
    if isinstance(config, dict):
        config_path = config["path"]
        overrrides = config["config_overrides"]
        model_args = init_hydra_config(config_path, overrides=overrrides)
        model = init_model(model_args.model.model_str, model_args.model.model_config)
    else:
        config_path = config
        model_args = init_hydra_config(config_path)
        model = init_model(model_args.model_str, model_args.model_config)
    model.to(device)
    if ckpt_path is not None:
        print("Loading model from: ", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        model.to(device)
    # Set the model to eval mode
    model.eval()

    return model


@torch.no_grad()
def sample_inference(model, views, device, use_amp):
    # Run inference
    result = loss_of_one_batch_multi_view(
        views,
        model,
        None,
        device,
        use_amp=use_amp,
    )

    return result
