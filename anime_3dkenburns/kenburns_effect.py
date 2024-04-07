from dataclasses import dataclass, fields,field
from typing import Any, Union, List, Optional
from copy import deepcopy
import numpy as np
import torchvision
import torch
from omegaconf import OmegaConf
import mmcv
import math
from PIL import Image
import os.path as osp
from pydensecrf.utils import compute_unary, unary_from_softmax
import pydensecrf.densecrf as dcrf
from tqdm import tqdm

from utils.effects import bokeh_blur
from animeinsseg import AnimeInsSeg, AnimeInstances
from animeinsseg.inpainting.ldm_inpaint import ldm_inpaint_webui
from animeinsseg.inpainting import patch_match
from animeinsseg.data.syndataset import quantize_image
from depth_modules import load_zoe, colorize, apply_leres
from .models import load_depth_refinenet, load_inpaintnet, disparity_estimation, pointcloud_inpainting, disparity_refinement
from .models.utils import spatial_filter, depth_to_points
from .common import process_autozoom, process_shift, render_pointcloud, fill_disocclusion

from utils.constants import DEFAULT_INPAINTNET_CKPT, DEFAULT_DEPTHREFINE_CKPT, DEFAULT_DETECTOR_CKPT, DEFAULT_DEVICE, DEPTH_ZOE_CKPT
from utils.io_utils import scaledown_maxsize
import cv2
import moviepy
from torchvision.transforms import GaussianBlur

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

UNIFORMER = None
    

def depth_adjustment_animesseg(instances: AnimeInstances, tenDisparity, tenImage, use_medium = False):
    assert(tenDisparity.shape[0] == 1)

    tenMasks = []
    if not instances.is_empty:
        for intMask in range(instances.masks.shape[0]):
            tenMasks.append(instances.masks[intMask].float())

    # for 

    if tenDisparity.shape[2] != tenImage.shape[2] or tenDisparity.shape[3] != tenImage.shape[3]:
        tenAdjusted = torch.nn.functional.interpolate(input=tenDisparity, size=(tenImage.shape[2], tenImage.shape[3]), mode='bilinear', align_corners=False)
        resized = True
    else:
        tenAdjusted = tenDisparity
        resized = False

    for tenAdjust in tenMasks:
        # msk = (tenAdjust.squeeze() * 255).cpu().numpy().astype(np.uint8)
        tenPlane = tenAdjusted * tenAdjust

        
        # if torch.any(torch.isnan(tenPlane)):
        #     print('shit', torch.any(torch.isnan(tenAdjust)), torch.any(torch.isnan(tenAdjusted)), \
        #           tenAdjusted.max(), tenAdjusted.min(), tenAdjust.max(), tenAdjust.min())

        # tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()
        # tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()

        if tenPlane.sum().item() == 0: continue

        # if torch.any(torch.isnan(tenPlane)):
        #     print('shit', torch.any(torch.isnan(tenAdjust)), torch.any(torch.isnan(tenAdjusted)), tpcp.max(), tpcp.min())

        if not use_medium:
            intLeft = (tenPlane.sum([2], True) > 0.0).flatten().nonzero()[0].item()
            intTop = (tenPlane.sum([3], True) > 0.0).flatten().nonzero()[0].item()
            intRight = (tenPlane.sum([2], True) > 0.0).flatten().nonzero()[-1].item()
            intBottom = (tenPlane.sum([3], True) > 0.0).flatten().nonzero()[-1].item()
            tenAdjusted = ((1.0 - tenAdjust) * tenAdjusted) + (tenAdjust * tenPlane[:, :, int(round(intTop + (0.97 * (intBottom - intTop)))):, :].max())
        else:
            tenAdjusted[tenPlane > 0] = tenAdjusted[tenPlane > 0].median()
        # torch.nan_to_num_(tenPlane, 0, 0, 0)
        
        # tenAdjusted = torch.clip(tenAdjusted, -100000, 100000)

    
    # end

    if resized:
        return torch.nn.functional.interpolate(input=tenAdjusted, size=(tenDisparity.shape[2], tenDisparity.shape[3]), mode='bilinear', align_corners=False)
    else:
        return tenAdjusted


def merge_mask(mask_dict1, mask_dict2):
    msk1, br1, area1 = mask_dict1
    w1, h1 = br1[2] - br1[0], br1[3] - br1[1]
    msk2, br2, area2 = mask_dict2
    w2, h2 = br2[2] - br2[0], br2[3] - br2[1]

    ax1, ay1, ax2, ay2 = br1
    bx1, by1, bx2, by2 = br2
    r = min(ax2, bx2)
    l = max(ax1, bx1)
    b = min(ay2, by2) 
    t = max(ay1, by1)
    if r <= l or b <= t:
        return None

    r = max(ax2, bx2)
    l = min(ax1, bx1)
    b = max(ay2, by2) 
    t = min(ay1, by1)
    w, h = r - l, b - t
    mask1 = np.zeros((h, w), np.uint8)
    mask2 = np.copy(mask1)
    x1, y1 = ax1 - l, ay1 - t
    mask1[y1: y1+h1, x1: x1+w1] = msk1
    x1, y1 = bx1 - l, by1 - t
    mask2[y1: y1+h2, x1: x1+w2] = msk2

    merge_thr = 0.1
    mand = np.bitwise_and(mask1, mask2).sum() / 255.
    merge_score = max(mand / area1, mand / area2)

    # cv2.imshow('msk1', msk1)
    # cv2.imshow('msk2', msk2)
    # cv2.imshow('mask1', mask1)
    # cv2.imshow('mask2', mask2)
    # cv2.imshow('merged', mask_merged)
    # print(merge_score)
    # cv2.waitKey(0)
    if merge_score > merge_thr:
        mask_merged =  np.bitwise_or(mask1, mask2)
        area = mask_merged.sum() / 255.
        # cv2.imshow('msk1', msk1)
        # cv2.imshow('msk2', msk2)
        # cv2.imshow('mask1', mask1)
        # cv2.imshow('mask2', mask2)
        # cv2.imshow('merged', mask_merged)
        # cv2.waitKey(0)
        return (mask_merged, [l, t, r, b], area), merge_score
    return None


# def try_merge_mask_list(mask_list: List):
#     valid_segs = [mask_list.pop(0)]
#     while len(mask_list) > 0:
#         seg = mask_list.pop(0)
#         merged = None
#         merge_score = -1
#         merge_idx = -1
#         for jj, tseg in enumerate(valid_segs):
#             merge_result = merge_mask(seg, tseg)
#             if merge_result is not None:
#                 if merge_result[1] > merge_score:
#                     merge_score = merge_result[1]
#                     merged = merge_result[0]
#                     merge_idx = jj
#         if merge_idx != -1:
#             valid_segs[merge_idx] = merged
#         else:
#             valid_segs.append(seg)
#     return valid_segs

def enlarge_window(rect, im_w, im_h, ratio=2.5, aspect_ratio=1.0):
    assert ratio > 1.0
    
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return [0, 0, 0, 0]

    # https://numpy.org/doc/stable/reference/generated/numpy.roots.html
    coeff = [aspect_ratio, w+h*aspect_ratio, (1-ratio)*w*h]
    roots = np.roots(coeff)
    roots.sort()
    delta = int(round(roots[-1] / 2 ))
    delta_w = int(delta * aspect_ratio)
    delta_w = min(x1, im_w - x2, delta_w)
    delta = min(y1, im_h - y2, delta)
    rect = np.array([x1-delta_w, y1-delta, x2+delta_w, y2+delta], dtype=np.int64)
    rect[::2] = np.clip(rect[::2], 0, im_w)
    rect[1::2] = np.clip(rect[1::2], 0, im_h)
    return rect.tolist()


def try_merge_mask_list(src_list: List, tgt_list: List, fin_list: List):
    while len(src_list) > 0:
        seg = src_list.pop(0)
        merged = None
        merge_score = -1
        merge_idx = -1
        for jj, tseg in enumerate(tgt_list):
            merge_result = merge_mask(seg, tseg)
            if merge_result is not None:
                if merge_result[1] > merge_score:
                    merge_score = merge_result[1]
                    merged = merge_result[0]
                    merge_idx = jj
        if merge_idx != -1:
            tgt_list[merge_idx] = merged
        else:
            fin_list.append(seg)

@dataclass
class KenBurnsConfig:

    # detector field
    detector: str = 'animeinsseg'
    det_ckpt: str = 'models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt'
    det_size: int = 640
    scale_depth: bool = False

    depth_field: bool = False

    # mask_refine_kwargs: dict = {}
    mask_refine_kwargs: dict = field(default_factory=dict)
    marigold_kwargs: dict = field(default_factory=dict)

    pred_score_thr: float = 0.3

    depth_est: str = 'zoe'
    depth_est_device: str = ''
    depth_refinement: str = 'default'
    depthest_use_medium: bool = False
    inpaint_type: str = 'default'
    
    # kenburns field
    num_frame: int = 75
    playback: bool = True
    auto_zoom: bool = True
    focal: float = 1024 / 2.0
    baseline: float = 40.0
    dof_speed: float = 50.

    depth_factor: int = 1
    lightness_factor: int = 13

    max_size: int = 720

    int_height: int = 1024
    int_width: int = 1024

    default_depth_refine: bool = False
    refine_crf: bool = True

    depth_est_size:int = 640
    disparity_min = 0
    disparity_max = 0
    depth_range = None
    tensor_raw_image = None
    original_img_nparray = None
    raw_disparity = None
    raw_depth = None
    raw_point = None
    raw_unaltered = None
    inpainted_img = None
    inpainted_disparity = None
    inpainted_depth = None
    inpainted_points = None

    # ldm_cfg = 'configs/guided_ldm_inpaint9_v15.yaml'
    # ldm_ckpt = 'models/AnimeInstanceSegmentation/anythingv4.5-inpainting.ckpt'
    # ldm_ckpt = 'F:/repos/stable-diffusion-webui/models/Stable-diffusion/sd-v1-5-inpainting.ckpt'
    sd_img2img_url: str = 'http://127.0.0.1:7860/sdapi/v1/img2img'
    ldm_inpaint_options: dict = field(default_factory=lambda: {
        'steps': 32,
        'cfg_scale': 7,
        'sample_name': 'DPM++ 2M Karras',
        'denoising_strength': 0.75,
        'inpainting_fill': 0,
        'seed': 0,
        'subseed': 0,
    })
    ldm_inpaint_size: int = 0
    bg_prompt = None

    instances: AnimeInstances = None

    save_path = r''

    stage_inpainted_imgs = []
    stage_inpainted_masks = []
    stage_depth_coarse = None
    stage_depth_adjusted = None
    stage_depth_final = None


    def __getitem__(self, item: str):
        if item == 'fltFocal':
            return self.focal
        elif item == 'fltBaseline':
            return self.baseline
        elif item == 'intWidth':
            return self.int_width
        elif item == 'intHeight':
            return self.int_height
        
        elif item == 'fltDispmin':
            return self.disparity_min
        elif item == 'fltDispmax':
            return self.disparity_max
        elif item == 'objDepthrange':
            return self.depth_range
        elif item == 'tenRawImage':
            return self.tensor_raw_image
        elif item == 'tenRawDisparity':
            return self.raw_disparity
        elif item == 'tenRawDepth':
            return self.raw_depth
        elif item == 'tenRawPoints':
            return self.raw_point
        elif item == 'tenRawUnaltered':
            return self.raw_unaltered
        elif item == 'tenInpaImage':
            return self.inpainted_img
        elif item == 'tenInpaDisparity':
            return self.inpainted_disparity
        elif item == 'tenInpaDepth':
            return self.inpainted_depth
        elif item == 'tenInpaPoints':
            return self.inpainted_points
        
        return getattr(self, item)
    
    def __setitem__(self, item, value):
        if item == 'fltFocal':
            self.focal = value
        elif item == 'fltBaseline':
            self.baseline = value
        elif item == 'intWidth':
            self.int_width = value
        elif item == 'intHeight':
            self.int_height = value
        
        elif item == 'fltDispmin':
            self.disparity_min = value
        elif item == 'fltDispmax':
            self.disparity_max = value
        elif item == 'objDepthrange':
            self.depth_range = value
        elif item == 'tenRawImage':
            self.tensor_raw_image = value
        elif item == 'tenRawDisparity':
            self.raw_disparity = value
        elif item == 'tenRawDepth':
            self.raw_depth = value
        elif item == 'tenRawPoints':
            self.raw_point = value
        elif item == 'tenRawUnaltered':
            self.raw_unaltered = value
        elif item == 'tenInpaImage':
            self.inpainted_img = value
        elif item == 'tenInpaDisparity':
            self.inpainted_disparity = value
        elif item == 'tenInpaDepth':
            self.inpainted_depth = value
        elif item == 'tenInpaPoints':
            self.inpainted_points = value
        else:
            setattr(self, item, value)

    def copy(self):
        return deepcopy(self)


def build_kenburns_cfg(tgt_cfg: Union[str, dict]):
    if isinstance(tgt_cfg, str):
        tgt_cfg = dict(OmegaConf.load(tgt_cfg))
    fieldSet = {f.name for f in fields(KenBurnsConfig) if f.init}
    filteredArgDict = {k : v for k, v in tgt_cfg.items() if k in fieldSet}
    return KenBurnsConfig(**filteredArgDict)


def save_disparity(disparity, save_path, cmap='magma_r'):
    cd = colorize(disparity.cpu().numpy(), cmap=cmap)
    Image.fromarray(cd).save(save_path)


def colorize_depth(depth, inverse=False, rgb2bgr=False, **kwargs):
    if inverse:
        depth = 1 / (depth + 1e-5)
    colored = colorize(depth, **kwargs)
    colored = colored[..., :3]
    if rgb2bgr:
        colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    return colored


class KenBurnsPipeline:

    def __init__(self, cfg: Union[KenBurnsConfig, str] = None, device: str = None) -> None:
        if cfg is None:
            cfg = self.cfg = KenBurnsConfig()
        elif isinstance(cfg, KenBurnsConfig):
            self.cfg = cfg
        elif isinstance(cfg, str) or isinstance(cfg, dict):
            self.cfg = cfg = build_kenburns_cfg(cfg)
        else:
            raise NotImplementedError

        self.animeinsseg = None
        self.maskrcnn = None
        self.depth_zoe = None
        self.mask_refinenet = None
        self.depth_refinenet = None
        self.kenburns_inpaintnet = None
        self.inpaint_type = 'default'
        self.ldm = None

        self.input_img = None
        if device is None:
            self.device = DEFAULT_DEVICE
        else:
            self.device = device

        self.set_detector(cfg.detector)
        self.set_depth_estimation(cfg.depth_est)
        if self.cfg.default_depth_refine:
            self.set_depth_refinement(cfg.depth_refinement)
        self.set_inpainting(cfg.inpaint_type)

    def set_inpainting(self, inpainting: str):
        self.inpaint_type = inpainting
        if self.kenburns_inpaintnet is None:
            self.kenburns_inpaintnet = load_inpaintnet(DEFAULT_INPAINTNET_CKPT, device=self.device)
        self._inpaint = lambda img_tensor, tensor_disparity, tensor_shift, cfg, segmasks: \
            pointcloud_inpainting(self.kenburns_inpaintnet, img_tensor, tensor_disparity, tensor_shift, cfg, segmasks)
            
        if inpainting == 'ldm':
        #     if self.ldm is None:
        #         ldm = create_model(self.cfg.ldm_cfg)
        #         load_ldm_sd(ldm, self.cfg.ldm_ckpt)
        #         self.ldm = ldm.to(self.device)
            if self.animeinsseg.tagger is None:
                    self.animeinsseg.init_tagger()
        # elif inpainting == 'patchmatch':
        
    def inpaint(self, tenShift, tenPoints, objCommon: KenBurnsConfig, verbose: bool = False):
        ins = objCommon.instances
        mask_with_ins = None
        if not ins.is_empty:
            mask_with_ins = ins.masks[0]
            if len(ins.masks) > 1:
                for m in ins.masks[1:]:
                    mask_with_ins = torch.logical_or(mask_with_ins, m)
            # mask_with_ins = torch.logical_or(mask_with_ins, (objInpainted['tenExisting'] == 0.0).squeeze())
            mask_with_ins = mask_with_ins.to(torch.float32)
            mask_with_ins = mask_with_ins.repeat(3, 1, 1).unsqueeze(0)

        objInpainted = self._inpaint(objCommon['tenRawImage'], objCommon['tenRawDisparity'], tenShift, objCommon, mask_with_ins)
        objInpainted['tenDepth'] = (objCommon['fltFocal'] * objCommon['fltBaseline']) / (objInpainted['tenDisparity'] + 0.0000001)
        objInpainted['tenValid'] = (spatial_filter(objInpainted['tenDisparity'] / objInpainted['tenDisparity'].max(), 'laplacian').abs() < 0.03).float()
        objInpainted['tenPoints'] = depth_to_points(objInpainted['tenDepth'] * objInpainted['tenValid'], objCommon['fltFocal'])
        objInpainted['tenPoints'] = objInpainted['tenPoints'].view(1, 3, -1)
        objInpainted['tenPoints'] = objInpainted['tenPoints'] - tenShift

        mask_with_ins = objInpainted['segmasks']

        tenMask = (objInpainted['tenExisting'] == 0.0).view(1, 1, -1)
        mask = tenMask.cpu().squeeze().numpy().reshape((objCommon.int_height, objCommon.int_width)).astype(np.uint8) * 255
        # cv2.imwrite('maskwithholes.png', mask)
        
        if mask_with_ins is not None:
            mask_with_ins = mask_with_ins[0, 0] > 0
            mask_with_ins = mask_with_ins.cpu().squeeze().numpy().reshape((objCommon.int_height, objCommon.int_width)).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, mask_with_ins)

        if self.inpaint_type == 'default':
            objCommon.inpainted_img = torch.cat([ objCommon.inpainted_img, 
                                                    objInpainted['tenImage'].view(1, 3, -1)[tenMask.repeat(1, 3, 1)].view(1, 3, -1) ], 2)
        elif self.inpaint_type == 'ldm':
            e_size = 5
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e_size + 1, 2 * e_size + 1),(e_size, e_size))
            mask_dilated = cv2.dilate(mask, element, iterations=1)

            original = objInpainted['tenImage'].cpu().numpy().squeeze()
            original = (original * 255).astype(np.uint8).transpose((1, 2, 0))

            prompt = self.get_bg_prompt(objCommon) + ', high quality, masterpiece, no_humans'
            neg_prompt = 'cat, human, single, person, girl, 1girl, creature, animal, alien, robot, body'
            resolution = self.cfg.ldm_inpaint_size if self.cfg.ldm_inpaint_size else self.cfg.max_size

            # cv2.imwrite('maskwithholes_original.png', original)
            
            print('running ldm inpainting ...')
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)   
            
            inpainted = ldm_inpaint_webui(original, mask_dilated, resolution, self.cfg.sd_img2img_url, prompt, neg_prompt, **self.cfg.ldm_inpaint_options)
            inpainted = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)
            inpainted = torch.FloatTensor(np.ascontiguousarray(inpainted.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).to(self.device)
            objCommon.inpainted_img = torch.cat([ objCommon.inpainted_img, inpainted.view(1, 3, -1)[tenMask.repeat(1, 3, 1)].view(1, 3, -1) ], 2)
            objInpainted['tenImage'] = inpainted

        elif self.inpaint_type == 'patchmatch':
            original = objInpainted['tenImage'].cpu().numpy().squeeze()
            original = (original * 255).astype(np.uint8).transpose((1, 2, 0))
            inpainted = patch_match.inpaint(original, mask, patch_size=3)
            inpainted = torch.FloatTensor(np.ascontiguousarray(inpainted.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).to(self.device)
            objCommon.inpainted_img = torch.cat([ objCommon.inpainted_img, inpainted.view(1, 3, -1)[tenMask.repeat(1, 3, 1)].view(1, 3, -1) ], 2)
            objInpainted['tenImage'] = inpainted
        
        inpainted = objInpainted['tenImage'].cpu().numpy().squeeze()
        inpainted = (inpainted * 255).astype(np.uint8).transpose((1, 2, 0))
        objCommon.stage_inpainted_imgs.append(inpainted)
        objCommon.stage_inpainted_masks.append(mask)

        objCommon['tenInpaDisparity'] = torch.cat([ objCommon['tenInpaDisparity'], objInpainted['tenDisparity'].view(1, 1, -1)[tenMask.repeat(1, 1, 1)].view(1, 1, -1) ], 2)
        objCommon['tenInpaDepth'] = torch.cat([ objCommon['tenInpaDepth'], objInpainted['tenDepth'].view(1, 1, -1)[tenMask.repeat(1, 1, 1)].view(1, 1, -1) ], 2)
        objCommon['tenInpaPoints'] = torch.cat([ objCommon['tenInpaPoints'], objInpainted['tenPoints'].view(1, 3, -1)[tenMask.repeat(1, 3, 1)].view(1, 3, -1) ], 2)
        
        # return tenMask

    def get_bg_prompt(self, kcfg: KenBurnsConfig):

        if kcfg.instances.is_empty:
            return ''
        
        if kcfg.bg_prompt is not None:
            return kcfg.bg_prompt

        mask = kcfg.instances.compose_masks('numpy').astype(np.uint8) * 255

        e_size = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e_size + 1, 2 * e_size + 1))
        mask = cv2.dilate(mask, element, iterations=1)
        inpaint_size = 448
        img = scaledown_maxsize(kcfg.original_img_nparray, inpaint_size)
        mask = scaledown_maxsize(mask, inpaint_size)

        inpainted = patch_match.inpaint(img, mask, patch_size=3)
        prompt = self.animeinsseg.tagger.label_cv2_bgr(inpainted)[0]
        prompt = ','.join(prompt)
        kcfg.bg_prompt = prompt
        return prompt
    

    def set_depth_estimation(self, depth_est: str = 'zoe'):
        if depth_est == 'zoe':
            if self.depth_zoe is None:
                self.depth_zoe = load_zoe(DEPTH_ZOE_CKPT, device=self.device, img_size=[672, 672])
            self._depth_est = self._depth_est_zoe
        elif depth_est == 'leres':
            self._depth_est = self._depth_est_leres
        elif depth_est == 'default':
            self._depth_est = lambda img_tensor, img, **kwargs : disparity_estimation(img_tensor)
        elif depth_est == 'marigold':
            self._depth_est = self._depth_marigold
        else:
            raise NotImplementedError(f'Invalid depth model: {depth_est}')
        
    def _depth_marigold(self, img_tensor, img, *args, **kwargs):
        from utils.apply_marigold import apply_marigold
        depth = apply_marigold(img, **self.cfg.marigold_kwargs)
        depth = depth.astype(np.float32)
        depth = torch.from_numpy(depth[None, None, ...]).to(self.device)
        depth[depth == 0] = depth[depth > 0].min()
        depth = (1 - depth) * 255
        return depth

    def _depth_est_leres(self, img_tensor, img, *args, **kwargs):
        
        device = self.cfg.depth_est_device
        if device == '':
            device = self.device

        ori_h, ori_w = img.shape[:2]
        img = scaledown_maxsize(img, max_size=self.cfg.depth_est_size, divisior=32)
        img = img.astype(np.float32) / 255.
        depth = apply_leres(img, device=device, boost=False)
        k = depth.shape[0] / ori_h
        depth = cv2.resize(depth, (ori_w, ori_h), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        depth = depth.astype(np.float32)
        depth = torch.from_numpy(depth[None, None, ...]).to(self.device)
        depth[depth == 0] = depth[depth > 0].min()
        # disparity = 1 / depth
        # disparity.nan_to_num_(0, 0, 0)
        print(depth.min(), depth.max(), depth.shape)
        return depth

    def infer_disparity(self, img: Union[np.ndarray, str], instances: AnimeInstances = None, img_tensor: torch.Tensor = None, save_dir: str = None, save_name: str = None, kcfg: KenBurnsConfig = None):
        
        save_vis = save_dir is not None and save_name is not None
        
        with torch.no_grad():
            if instances is None:
                instances, img = self.run_instance_segmentation(img, scale_down_to_maxsize=False)
                torch_gc()

            # coarse depth
            if img_tensor is None:
                img_tensor = torch.FloatTensor(np.ascontiguousarray(img.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).to(self.device)
            disparity = self._depth_est(img_tensor, img)
            
            depth_coarse_vis = colorize_depth(disparity.cpu().numpy(), inverse=True, rgb2bgr=True, cmap='magma_r')
            if kcfg is not None:
                kcfg.stage_depth_coarse = depth_coarse_vis
            if save_vis:
                Image.fromarray(depth_coarse_vis).save(osp.join(save_dir, 'tmp_stage_depth_coarse_' + save_name+'.png'))

            # adjusted depth
            disparity = depth_adjustment_animesseg(instances, disparity, img_tensor, self.cfg.depthest_use_medium)

            depth_adjusted_vis = colorize_depth(disparity.cpu().numpy(), inverse=True, rgb2bgr=True, cmap='magma_r')
            if kcfg is not None:
                kcfg.stage_depth_adjusted = depth_adjusted_vis
            if save_vis:
                Image.fromarray(depth_adjusted_vis).save(osp.join(save_dir, 'tmp_stage_depth_adjusted_' + save_name+'.png'))

            # if self.cfg.scale_depth:
            #     disparity, sd_img = self.scale_depth(img_tensor, instances, disparity, img)
            #     if save_vis:
            #         save_disparity(disparity, osp.join(save_dir, save_name+'-disparity_scaled.png'))
            #         cv2.imwrite(osp.join(save_dir, save_name+'-sd_for_depth.png'), sd_img)

            # final depth
            if self.cfg.default_depth_refine:
                disparity = self.refine_depth(img_tensor, disparity)
            elif self.cfg.refine_crf:
                disparity = self.refine_depth_crf(img, disparity, instances)

            # g = GaussianBlur((63, 63))
            # disparity = g(disparity)
            depth_final_vis = colorize_depth(disparity.cpu().numpy(), inverse=True, rgb2bgr=True, cmap='magma_r')
            # cv2.imwrite('depth.png', depth_final_vis[..., 0])
            if kcfg is not None:
                kcfg.stage_depth_final = depth_final_vis
            if save_vis:
                Image.fromarray(depth_final_vis).save(osp.join(save_dir, 'tmp_stage_depth_final_' + save_name+'.png'))

            return disparity
        
        
    def refine_depth_crf(self, img: np.ndarray, disparity: torch.Tensor, instances: AnimeInstances):
        # quantize_image(7, )
    
        def crf_refine(rawmask, rgbimg):
            if len(rawmask.shape) == 2:
                rawmask = rawmask[:, :, None]
            mask_softmax = np.concatenate([cv2.bitwise_not(rawmask)[:, :, None], rawmask], axis=2)
            mask_softmax = mask_softmax.astype(np.float32) / 255.0
            n_classes = 2
            feat_first = mask_softmax.transpose((2, 0, 1)).reshape((n_classes,-1))
            unary = unary_from_softmax(feat_first)
            unary = np.ascontiguousarray(unary)

            d = dcrf.DenseCRF2D(rgbimg.shape[1], rgbimg.shape[0], n_classes)

            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=1, compat=3, kernel=dcrf.DIAG_KERNEL,
                                    normalization=dcrf.NO_NORMALIZATION)

            d.addPairwiseBilateral(sxy=46, srgb=4, rgbim=rgbimg,
                                compat=40,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NO_NORMALIZATION)
            # d.addPairwiseBilateral(sxy=46, srgb=14, rgbim=rgbimg,
            #                     compat=20,
            #                     kernel=dcrf.DIAG_KERNEL,
            #                     normalization=dcrf.NO_NORMALIZATION)
            Q = d.inference(5)
            res = np.argmax(Q, axis=0).reshape((rgbimg.shape[0], rgbimg.shape[1]))
            crf_mask = np.array(res * 255, dtype=np.uint8)
            return crf_mask        

        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_h, im_w = rgbimg.shape[:2]
        img_size = im_h * im_w
        ins_mask = instances.compose_masks(output_type='numpy')
        if ins_mask is not None:
            ins_mask = np.logical_not(ins_mask)
        depth_npy = colorize(disparity.cpu().numpy(), cmap='gray_r')[..., 0]
        depth_npy = np.ascontiguousarray(depth_npy[..., None])
        depth_quantized, centers, labels = quantize_image(depth_npy, 5, mask=ins_mask)

        # detected_edges = cv2.Canny(cv2.GaussianBlur(depth_quantized, ksize=(5, 5), sigmaX=1, sigmaY=1), 50, 140, L2gradient=True, apertureSize=3)
        detected_edges = cv2.Canny(depth_npy, 50, 140, L2gradient=True, apertureSize=3)
        # detected_edges = cv2.bitwise_or(detected_edges, edges)
        e_size = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e_size + 1, 2 * e_size + 1),(e_size, e_size))
        detected_edges = cv2.morphologyEx(detected_edges,cv2.MORPH_CLOSE, element, iterations = 1)

        depth_npy[detected_edges > 0] = 255 - depth_npy[detected_edges > 0]
        centers = [c[0] for c in centers]
        centers.sort()

        if ins_mask is not None:
            ins_mask = np.ascontiguousarray(ins_mask[..., None])

        size_thr = max(img_size / 1000, 5)

        hier_segments = []
        for c in centers:
            hier_segments.append([])
            c = int(c * 255)
            rawmask = (depth_quantized == c)
            if ins_mask is not None:
                rawmask = np.logical_and(rawmask, ins_mask)
            rawmask = rawmask.squeeze().astype(np.uint8) * 255
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(rawmask, connectivity=4)
            # cv2.imshow('rawmask', rawmask)
            for label_index, stat, centroid in zip(range(num_labels), stats, centroids):
                if label_index != 0: # skip background label
                    x, y, w, h, area = stat
                    if w * h < size_thr:
                        continue
                    x1, y1, x2, y2 = x, y, x+w, y+h
                    label_local = labels[y1: y2, x1: x2]
                    label_cordinates = np.where(label_local==label_index)
                    local_mask = np.zeros_like(label_local, np.uint8)
                    local_mask[label_cordinates] = 255
                    cv2.rectangle(local_mask, (0, 0), (local_mask.shape[1], local_mask.shape[0]), 0, 1)
                    dist = cv2.distanceTransform(local_mask, distanceType=cv2.DIST_L2, maskSize=3)
                    dist = (dist * 255. / dist.max()).astype(np.uint8)
                    seed = np.argmax(dist)
                    seedy, seedx = np.unravel_index(seed, dist.shape)

                    # seed_pnts.append([seedx + x1, seedy + y1])
                    # local_mask = cv2.circle(local_mask, (seedx, seedy), 10, (0, 0, 0), 5)

                    ex1, ey1, ex2, ey2 = enlarge_window([x1, y1, x2, y2], im_w, im_h, ratio=4)
                    # ex1, ey1, ex2, ey2 = 0, 0, im_w, im_h
                    # print(ex1, ey1, ex2, ey2)
                    seedx = seedx - ex1 + x1
                    seedy = seedy - ey1 + y1
                    dc = depth_npy[ey1: ey2, ex1: ex2].copy()
                    seed_val = dc[seedy, seedx]

                    fillmsk = np.zeros((dc.shape[0] + 2, dc.shape[1] + 2), dtype=np.uint8)
                    if ins_mask is not None:
                        # fillmsk[1:-1, 1:-1] = np.logical_xor(fillmsk[1:-1, 1:-1], ins_mask[ey1: ey2, ex1: ex2, 0])
                        fillmsk[1:-1, 1:-1] = np.logical_not(ins_mask[ey1: ey2, ex1: ex2, 0])
                        # cv2.imshow('fillmsk', fillmsk)
                    # dc = cv2.circle(dc, (seedx, seedy), 10, (255, 0, 0), 5)
                    fdiff = 4
                    ret, im_out, msk_out, _ = cv2.floodFill(dc, mask=fillmsk, seedPoint=(seedx, seedy), newVal=255, loDiff=fdiff, upDiff=fdiff, flags=cv2.FLOODFILL_MASK_ONLY | 4)
                    msk_out *= 255
                    msk_out = np.ascontiguousarray(msk_out[1:-1, 1:-1])
                    e_size = 1
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e_size + 1, 2 * e_size + 1),(e_size, e_size))
                    msk_out = cv2.dilate(msk_out, kernel=k, iterations=1)
                    if ins_mask is not None:
                        msk_out[ins_mask[ey1: ey2, ex1: ex2, 0] == 0] = 0
                    rgbc = np.ascontiguousarray(rgbimg[ey1: ey2, ex1: ex2])
                    crf = crf_refine(msk_out, rgbc)
                    # crf = cv2.bitwise_or(crf, cv2.erode(msk_out, kernel=element, iterations=1))
                    crf = cv2.morphologyEx(crf,cv2.MORPH_CLOSE, element, iterations = 1)
                    x1, y1, w, h = cv2.boundingRect(cv2.findNonZero(crf))
                    if w * h < size_thr:
                        continue
                    crf = crf[y1: y1+h, x1: x1+w]
                    x1 += ex1
                    y1 += ey1
                    
                    if msk_out.sum() / ((crf.sum() + 0.1)) > 0.5:
                        hier_segments[-1].append([crf, [x1, y1, w+x1, h+y1], ret])

        
        if len(hier_segments) > 0:

            final_segs = []
            for ii, segments in enumerate(hier_segments[:-1]):
                if len(segments) == 0:
                    continue
                src_segs = [segments.pop(0)]
                try_merge_mask_list(segments, src_segs, src_segs)
                try_merge_mask_list(src_segs, hier_segments[ii+1], final_segs)

            last_layer_segs = hier_segments[-1]
            if len(last_layer_segs) > 0:
                src_segs = [last_layer_segs.pop(0)]
                try_merge_mask_list(last_layer_segs, src_segs, src_segs)
                final_segs += src_segs

            disparity_cpu = disparity.cpu()
            if ins_mask is not None:
                ins_mask = ins_mask.squeeze().astype(np.uint8) * 255
            for seg in final_segs:
                mask, br, area = seg
                e_size = 2
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e_size + 1, 2 * e_size + 1),(e_size, e_size))
                
                mask = cv2.dilate(mask, kernel=k, iterations=1)
                if ins_mask is not None:    
                    mask = np.bitwise_and(mask, ins_mask[br[1]: br[3], br[0]: br[2]])
                    ins_mask[br[1]: br[3], br[0]: br[2]][mask > 0] = 0
                if mask.sum() / 255 > img_size / 10:
                    continue
                mask_tensor = torch.from_numpy(mask > 0)
                refine_mean = disparity_cpu[0, 0, br[1]: br[3], br[0]: br[2]][mask_tensor].median()
                disparity_cpu[0, 0, br[1]: br[3], br[0]: br[2]][mask_tensor] = refine_mean
                
                # cv2.imshow('final mask', mask)
                # # d = depth_npy[br[1]: br[3], br[0]: br[2]]
                # # cv2.imshow('d', d)
                # cv2.waitKey(0)
            disparity = disparity_cpu.to(self.device)


        # cv2.imshow('depth', depth_npy)
        # cv2.imshow('insmask', ins_mask.astype(np.uint8) * 255)
        # cv2.imshow('depth quantized', depth_quantized)
        # cv2.imshow('canny', detected_edges)
        # cv2.waitKey(0)
        return disparity


    def _depth_est_zoe(self, img_tensor, *args, **kwargs):
        depth = self.depth_zoe.infer(img_tensor, with_flip_aug=True, pad_input=True)

        depth[depth == 0] = depth[depth > 0].min()
        disparity = (self.cfg.focal * self.cfg.baseline) / (depth + 0.00001)
        disparity.nan_to_num_(0, 0, 0)
        return disparity.to(self.device)

    def set_depth_refinement(self, depth_refinement: str):
        if depth_refinement == 'default':
            if self.depth_refinenet is None:
                self.depth_refinenet = load_depth_refinenet(DEFAULT_DEPTHREFINE_CKPT, device=self.device)
            self._refine_depth = lambda img, disparity: disparity_refinement(self.depth_refinenet, img, disparity)
        else:
            raise NotImplementedError(f'Invalid depth refinement: {depth_refinement}')

    def refine_depth(self, img: torch.Tensor, disparity: torch.Tensor):
        return self._refine_depth(img, disparity)

    def set_detector(self, detector: str = 'maskrcnn'):
        if detector == 'animeinsseg':
            if self.animeinsseg is None:
                self.animeinsseg = AnimeInsSeg(self.cfg.det_ckpt, device=self.device)
            # self.animeinsseg.set_mask_threshold(0.2)
            # self.animeinsseg.set_detect_size(self.cfg.det_size)
            self._instence_forward = self.animeinsseg_forward
        elif detector == 'maskrcnn' and self.maskrcnn is None:
            self.maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval().to(self.device)
            self._instence_forward = self.maskrcnn_forward
        else:
            raise NotImplementedError(f'Invalid detector: {detector}')

    def run_instance_segmentation(self, img: np.ndarray, scale_down_to_maxsize: bool = True):
        if scale_down_to_maxsize:
            img = scaledown_maxsize(img, self.cfg.max_size)
        with torch.no_grad():
            instances = self._instence_forward(img)
        return instances, img

    def animeinsseg_forward(self, img: np.ndarray, img_tensor: torch.Tensor = None):
        instances = self.animeinsseg.infer(img, self.cfg.pred_score_thr, \
                                               self.cfg.mask_refine_kwargs, output_type='tensor')
        return instances

    def maskrcnn_forward(self, img: np.ndarray, img_tensor: torch.Tensor = None) -> torch.Tensor:
        if img_tensor is None:
            img_tensor = torch.FloatTensor(np.ascontiguousarray(img.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).to(self.device)
        objPredictions = self.maskrcnn([ img_tensor[ 0, [ 2, 0, 1 ], :, : ] ])[0]
        valid_ids = objPredictions['scores'] > 0.3
        if valid_ids.sum() == 0:
            return AnimeInstances()
        
        boxes = objPredictions['boxes'][valid_ids]
        masks = objPredictions['masks'][valid_ids][:, 0]
        masks = masks > 0.5
        scores = objPredictions['scores'][valid_ids]
        return AnimeInstances(masks, boxes, scores)
    
    def groundedsam_forward(self, img: np.ndarray, img_tensor: torch.Tensor = None):
        
        pass

    def set_config(self, cfg: KenBurnsConfig):
        self.cfg = cfg

    def update_config_param(self, cfg_key: str, cfg_value: Any):
        self.cfg[cfg_key] = cfg_value

    def generate_kenburns_config(self, img: np.ndarray, instances: Optional[AnimeInstances] = None, verbose: bool = False, savep=None):
        '''Generate Kenburns Configuration for input image. 
        Returned config contains depth info and rendered point cloud of input.
        
        Args:
            img (str, ndarray, Sequence[str/ndarray]):
                Either image file or loaded images.
            instances (InstanceData): Optional, instance data of image
        
        '''

        if isinstance(img, str):
            img = mmcv.imread(img)

        with torch.no_grad():
            # img = scaledown_maxsize(img, self.cfg.max_size)
            if instances is None:
                instances, _ = self.run_instance_segmentation(img, scale_down_to_maxsize=False)
            
            img = scaledown_maxsize(img, self.cfg.max_size)
            instances.resize(img.shape[0], img.shape[1])
            self.cfg.int_height, self.cfg.int_width = img.shape[:2]

            img_tensor = torch.FloatTensor(np.ascontiguousarray(img.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).to(self.device)

            cfg: KenBurnsConfig = self.cfg.copy()
            disparity = self.infer_disparity(img, instances, img_tensor, kcfg=cfg)

            torch_gc()

            disparity = disparity / disparity.max() * self.cfg.baseline
            depth = (cfg.focal * cfg.baseline) / (disparity + 0.00001)

            tenValid = (spatial_filter(disparity / disparity.max(), 'laplacian').abs() < 0.03).float()
            tenPoints = depth_to_points(depth * tenValid, cfg.focal)
            tenUnaltered = depth_to_points(depth, cfg.focal)

            cfg['fltDispmin'] = disparity.min().item()
            cfg['fltDispmax'] = disparity.max().item()
            cfg['objDepthrange'] = cv2.minMaxLoc(src=depth[0, 0, 128:-128, 128:-128].detach().cpu().numpy(), mask=None)
            cfg['tenRawImage'] = img_tensor
            cfg['tenRawDisparity'] = disparity
            cfg['tenRawDepth'] = depth
            cfg['tenRawPoints'] = tenPoints.view(1, 3, -1)
            cfg['tenRawUnaltered'] = tenUnaltered.view(1, 3, -1)

            cfg.inpainted_img = cfg['tenRawImage'].view(1, 3, -1)
            cfg['tenInpaDisparity'] = cfg['tenRawDisparity'].view(1, 1, -1)
            cfg['tenInpaDepth'] = cfg['tenRawDepth'].view(1, 1, -1)
            cfg['tenInpaPoints'] = cfg['tenRawPoints'].view(1, 3, -1)

            cfg.instances = instances
            cfg.original_img_nparray = img
            return cfg
        
    def autozoom(self, cfg: KenBurnsConfig, verbose: bool = False):
        # process_autozoom(cfg)
        with torch.no_grad():
            objFrom = {
                'fltCenterU': cfg.int_width / 2.0,
                'fltCenterV': cfg.int_height / 2.0,
                'intCropWidth': int(math.floor(0.97 * cfg.int_width)),
                'intCropHeight': int(math.floor(0.97 * cfg.int_height))
            }

            objTo = process_autozoom({
                'fltShift': 100.0,
                'fltZoom': 1.25,
                'objFrom': objFrom
            }, cfg)

            # Debug by Francis
            npy_frame_list,_ = self.process_kenburns({
                'fltSteps': np.linspace(0.0, 1.0, cfg.num_frame).tolist(),
                'objFrom': objFrom,
                'objTo': objTo,
                'boolInpaint': True
            }, cfg, True, verbose)

            return npy_frame_list
    
    def process_kenburns(self, objSettings, objCommon: KenBurnsConfig, inpaint: bool = True, verbose: bool = False):
        with torch.no_grad():
            torch_gc()
            frames = []

            if inpaint:
                objCommon.inpainted_img = objCommon['tenRawImage'].view(1, 3, -1)
                objCommon['tenInpaDisparity'] = objCommon['tenRawDisparity'].view(1, 1, -1)
                objCommon['tenInpaDepth'] = objCommon['tenRawDepth'].view(1, 1, -1)
                objCommon['tenInpaPoints'] = objCommon['tenRawPoints'].view(1, 3, -1)

                for fltStep in [ 0.0, 1.0 ]:
                    fltFrom = 1.0 - fltStep
                    fltTo = 1.0 - fltFrom

                    fltShiftU = ((fltFrom * objSettings['objFrom']['fltCenterU']) + (fltTo * objSettings['objTo']['fltCenterU'])) - (objCommon['intWidth'] / 2.0)
                    fltShiftV = ((fltFrom * objSettings['objFrom']['fltCenterV']) + (fltTo * objSettings['objTo']['fltCenterV'])) - (objCommon['intHeight'] / 2.0)
                    fltCropWidth = (fltFrom * objSettings['objFrom']['intCropWidth']) + (fltTo * objSettings['objTo']['intCropWidth'])
                    fltCropHeight = (fltFrom * objSettings['objFrom']['intCropHeight']) + (fltTo * objSettings['objTo']['intCropHeight'])

                    fltDepthFrom = objCommon['objDepthrange'][0]
                    fltDepthTo = objCommon['objDepthrange'][0] * (fltCropWidth / max(objSettings['objFrom']['intCropWidth'], objSettings['objTo']['intCropWidth']))

                    shift = process_shift({
                        'tenPoints': objCommon['tenInpaPoints'],
                        'fltShiftU': fltShiftU,
                        'fltShiftV': fltShiftV,
                        'fltDepthFrom': fltDepthFrom,
                        'fltDepthTo': fltDepthTo
                    }, objCommon)
                    tenShift = shift[1]
                    tenPoints = shift[0]

                    self.inpaint(1.1 * tenShift, tenPoints, objCommon, verbose)


            for frame_idx, fltStep in enumerate(tqdm(objSettings['fltSteps'])):

                fltFrom = 1.0 - fltStep
                fltTo = 1.0 - fltFrom

                fltShiftU = ((fltFrom * objSettings['objFrom']['fltCenterU']) + (fltTo * objSettings['objTo']['fltCenterU'])) - (objCommon['intWidth'] / 2.0)
                fltShiftV = ((fltFrom * objSettings['objFrom']['fltCenterV']) + (fltTo * objSettings['objTo']['fltCenterV'])) - (objCommon['intHeight'] / 2.0)
                fltCropWidth = (fltFrom * objSettings['objFrom']['intCropWidth']) + (fltTo * objSettings['objTo']['intCropWidth'])
                fltCropHeight = (fltFrom * objSettings['objFrom']['intCropHeight']) + (fltTo * objSettings['objTo']['intCropHeight'])

                fltDepthFrom = objCommon['objDepthrange'][0]
                fltDepthTo = objCommon['objDepthrange'][0] * (fltCropWidth / max(objSettings['objFrom']['intCropWidth'], objSettings['objTo']['intCropWidth']))

                tenPoints = process_shift({
                    'tenPoints': objCommon['tenInpaPoints'],
                    'fltShiftU': fltShiftU,
                    'fltShiftV': fltShiftV,
                    'fltDepthFrom': fltDepthFrom,
                    'fltDepthTo': fltDepthTo
                }, objCommon)[0]

                inpainted = objCommon.inpainted_img
                tenRender, tenExisting = render_pointcloud(tenPoints, torch.cat([ inpainted, objCommon['tenInpaDepth'] ], 1).view(1, 4, -1), objCommon['intWidth'], objCommon['intHeight'], objCommon['fltFocal'], objCommon['fltBaseline'])

                tenRender = fill_disocclusion(tenRender, tenRender[:, 3:4, :, :] * (tenExisting > 0.0).float())
                frame = (tenRender[0, 0:3, :, :].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0.0, 255.0).astype(np.uint8)

                if objCommon.depth_field:
                    depth_rendered = tenRender[0, 3, :, :].detach().cpu().squeeze().numpy()
                    depth_rendered = colorize(depth_rendered, cmap='gray_r')[..., 0]
                    if frame_idx == 0:
                        focalplane_start = 0
                        focalplane_end = 255
                        ins = objCommon.instances
                        if not ins.is_empty:
                            focalplane_end = -1
                            for mask in ins.masks:
                                mask = mask.cpu().numpy()
                                dm = np.median(depth_rendered[mask])
                                if dm > focalplane_end:
                                    focalplane_end = dm
                            if abs(255 - focalplane_end) > abs(0 - focalplane_end):
                                focalplane_start = 255
                            else:
                                focalplane_start = 0

                    lightness_factor = objCommon.lightness_factor      
                    depth_num_samples = 32
                    depth_factor = objCommon.depth_factor

                    focal_int = 1/(1 + np.exp((0.5-fltStep) * objCommon.dof_speed))
                    focal_plane = focal_int * focalplane_end + (1-focal_int) * focalplane_start
                    frame = bokeh_blur(frame, depth_rendered, depth_num_samples, lightness_factor, focal_plane=focal_plane, use_cuda=True, depth_factor=depth_factor)

                frame = cv2.getRectSubPix(image=frame, patchSize=(max(objSettings['objFrom']['intCropWidth'], objSettings['objTo']['intCropWidth']), max(objSettings['objFrom']['intCropHeight'], objSettings['objTo']['intCropHeight'])), center=(objCommon['intWidth'] / 2.0, objCommon['intHeight'] / 2.0))
                frame = cv2.resize(src=frame, dsize=(objCommon['intWidth'], objCommon['intHeight']), fx=0.0, fy=0.0, interpolation=cv2.INTER_LINEAR)

                frames.append(frame)
            # end
            torch_gc()

            # Debug by Francis
            return\
            [
                frames,
                objCommon,
            ]

    
import moviepy.editor

def npyframes2video(npy_frame_list: List[np.ndarray], video_save_path: str, playback: bool = False):
    sequence = [npyFrame[:, :, ::-1] for npyFrame in npy_frame_list]
    if playback:
        sequence += sequence[::-1][1:-1]
    moviepy.editor.ImageSequenceClip(sequence=sequence, fps=25).write_videofile(video_save_path, preset="placebo")

