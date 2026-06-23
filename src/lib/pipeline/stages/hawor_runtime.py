"""Runner construction for the HaWoR model and the infiller network from their checkpoints."""

from pathlib import Path
from contextlib import contextmanager

import torch

from infiller.lib.model.network import TransformerModel
from lib.models.hawor import HAWOR
from lib.pipeline.hands.mano_runtime import resolve_mano_model_dir

from .hawor_common import vprint


@contextmanager
def _trusted_torch_load_defaults(*, weights_only):
    original_load = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", weights_only)
        return original_load(*args, **kwargs)

    torch.load = _load
    try:
        yield
    finally:
        torch.load = original_load


def load_hawor(checkpoint_path):
    from hawor.configs import get_config

    model_cfg = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
    model_cfg = get_config(model_cfg, update_cachedir=True)

    resolved_mano_dir = resolve_mano_model_dir(is_right=True)
    model_cfg.defrost()
    model_cfg.MANO.MODEL_PATH = str(resolved_mano_dir)
    model_cfg.MANO.DATA_DIR = str(resolved_mano_dir.parent)
    model_cfg.MANO.MEAN_PARAMS = str(resolved_mano_dir.parent / "mano_mean_params.npz")
    # EgoSmith: never torch.compile the backbone. The stock HaWoR __init__ compiles
    # when this flag is set; we disable it here (config override) instead of patching
    # the obtained hawor.py, so that file stays an unmodified upstream symlink.
    if "TORCH_COMPILE" in model_cfg.MODEL.BACKBONE:
        model_cfg.MODEL.BACKBONE.TORCH_COMPILE = 0
    model_cfg.freeze()

    if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, (
            f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        )
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    with _trusted_torch_load_defaults(weights_only=False):
        model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg


def build_motion_runner(checkpoint_path, device=None):
    model, model_cfg = load_hawor(checkpoint_path)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model.backbone = torch.compile(model.backbone, mode="reduce-overhead")
            vprint("[torch.compile] ViT backbone compiled with mode='reduce-overhead'")
        except Exception as error:
            vprint(f"[torch.compile] Skipping backbone compilation: {error}")
    return {
        "model": model,
        "model_cfg": model_cfg,
        "device": device,
    }


def build_infiller_runner(weight_path, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    pos_dim = 3
    shape_dim = 10
    num_joints = 15
    rot_dim = (num_joints + 1) * 6
    repr_dim = 2 * (pos_dim + shape_dim + rot_dim)
    horizon = 120
    filling_model = TransformerModel(
        seq_len=horizon,
        input_dim=repr_dim,
        d_model=384,
        nhead=8,
        d_hid=2048,
        nlayers=8,
        dropout=0.05,
        out_dim=repr_dim,
        masked_attention_stage=True,
    )
    filling_model.to(device)
    filling_model.load_state_dict(checkpoint["transformer_encoder_state_dict"])
    filling_model.eval()
    return {
        "model": filling_model,
        "device": device,
        "horizon": horizon,
        "src_mask": torch.zeros((horizon, horizon), device=device, dtype=torch.bool),
    }
