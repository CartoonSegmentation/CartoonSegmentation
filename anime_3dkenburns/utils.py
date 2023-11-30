import matplotlib
import matplotlib.cm
import numpy as np
import torch
import torch.nn
import torch.nn as nn

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
# import torch
# from tqdm import tqdm
# from einops import rearrange
# import cv2
# import numpy as np
# import os
# import os.path as osp
# from pathlib import Path
# from zoedepth.utils.misc import colorize


def load_zoe(ckpt_path: str, device: str = None, img_size=[512, 672]):

    conf = get_config("zoedepth", "infer")
    conf['pretrained_resource'] = 'local::'+ckpt_path
    conf['img_size'] = img_size
    model = build_model(conf)
    model.eval().to(device)
    return model