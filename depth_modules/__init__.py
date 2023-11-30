import matplotlib
import matplotlib.cm
import numpy as np
import torch
import torch.nn

from importlib import import_module

from .zoedepth.utils.misc import colorize
from .zoedepth.models.depth_model import DepthModel
from .zoedepth.utils.config import get_config
from .leres import apply_leres

def build_model(config) -> DepthModel:
    """Builds a model from a config. The model is specified by the model name and version in the config. The model is then constructed using the build_from_config function of the model interface.
    This function should be used to construct models for training and evaluation.

    Args:
        config (dict): Config dict. Config is constructed in utils/config.py. Each model has its own config file(s) saved in its root model folder.

    Returns:
        torch.nn.Module: Model corresponding to name and version as specified in config
    """
    module_name = f"depth_modules.zoedepth.models.{config.model}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as e:
        # print the original error message
        print(e)
        raise ValueError(
            f"Model {config.model} not found. Refer above error for details.") from e
    try:
        get_version = getattr(module, "get_version")
    except AttributeError as e:
        raise ValueError(
            f"Model {config.model} has no get_version function.") from e
    return get_version(config.version_name).build_from_config(config)


def load_zoe(ckpt_path: str, device: str = None, img_size=[512, 672]):

    conf = get_config("zoedepth", "infer")
    conf['pretrained_resource'] = 'local::'+ckpt_path
    conf['img_size'] = img_size
    model = build_model(conf)
    model.eval().to(device)
    return model