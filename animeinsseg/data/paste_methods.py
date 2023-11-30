import numpy as np
from typing import List, Union, Tuple, Dict
import random
from PIL import Image
import cv2
import os.path as osp
from tqdm import tqdm
from panopticapi.utils import rgb2id, id2rgb 
from time import time
import traceback

from utils.io_utils import bbox_overlap_area
from utils.logger import LOGGER
from utils.constants import COLOR_PALETTE



class PartitionTree:

    def __init__(self, bleft: int, btop: int, bright: int, bbottom: int, parent = None) -> None:
        self.left: PartitionTree = None
        self.right: PartitionTree = None
        self.top: PartitionTree = None
        self.bottom: PartitionTree = None

        if bright < bleft:
            bright = bleft
        if bbottom < btop:
            bbottom = btop

        self.bleft = bleft
        self.bright = bright
        self.btop = btop
        self.bbottom = bbottom
        self.parent: PartitionTree = parent

    def is_leaf(self):
        return self.left is None

    def new_partition(self, new_rect: List):
        self.left = PartitionTree(self.bleft, self.btop, new_rect[0], self.bbottom, self)
        self.top = PartitionTree(self.bleft, self.btop, self.bright, new_rect[1], self)
        self.right = PartitionTree(new_rect[2], self.btop, self.bright, self.bbottom, self)
        self.bottom = PartitionTree(self.bleft, new_rect[3], self.bright, self.bbottom, self)
        if self.parent is not None:
            self.root_update_rect(new_rect)

    def root_update_rect(self, rect):
        root = self.get_root()
        root.update_child_rect(rect)

    def update_child_rect(self, rect: List):
        if self.is_leaf():
            self.update_from_rect(rect)
        else:
            self.left.update_child_rect(rect)
            self.right.update_child_rect(rect)
            self.top.update_child_rect(rect)
            self.bottom.update_child_rect(rect)

    def get_root(self):
        if self.parent is not None:
            return self.parent.get_root()
        else:
            return self
                

    def update_from_rect(self, rect: List):
        if not self.is_leaf():
            return
        ix = min(self.bright, rect[2]) - max(self.bleft, rect[0])
        iy = min(self.bbottom, rect[3]) - max(self.btop, rect[1])
        if not (ix > 0  and iy > 0):
            return

        new_ltrb0 = np.array([self.bleft, self.btop, self.bright, self.bbottom])
        new_ltrb1 = new_ltrb0.copy()

        if rect[0] > self.bleft and rect[0] < self.bright:
            new_ltrb0[2] = rect[0]
        else:
            new_ltrb0[0] = rect[2]

        if rect[1] > self.btop and rect[1] < self.bbottom:
            new_ltrb1[3]= rect[1]
        else:
            new_ltrb1[1] = rect[3]

        if (new_ltrb0[2:] - new_ltrb0[:2]).prod() > (new_ltrb1[2:] - new_ltrb1[:2]).prod():
            self.bleft, self.btop, self.bright, self.bbottom = new_ltrb0
        else:
            self.bleft, self.btop, self.bright, self.bbottom = new_ltrb1

    @property
    def width(self) -> int:
        return self.bright  - self.bleft
    
    @property
    def height(self) -> int:
        return self.bbottom -  self.btop

    def prefer_partition(self, tgt_h: int, tgt_w: int):
        if self.is_leaf():
            return self, min(self.width / tgt_w, 1.2) * min(self.height / tgt_h, 1.2)
        else:
            lp, ls = self.left.prefer_partition(tgt_h, tgt_w)
            rp, rs = self.right.prefer_partition(tgt_h, tgt_w)
            tp, ts = self.top.prefer_partition(tgt_h, tgt_w)
            bp, bs = self.bottom.prefer_partition(tgt_h, tgt_w)
            preferp = [(p, s) for s, p in sorted(zip([ls, rs, ts, bs],[lp, rp, tp, bp]), key=lambda pair: pair[0], reverse=True)][0]
            return preferp

    def new_random_pos(self, fg_h: int, fg_w: int, im_h: int, im_w: int, random_sample: bool = False):
        extx, exty = int(fg_w / 3), int(fg_h / 3)
        extxb, extyb = int(fg_w / 10), int(fg_h / 10)
        region_w, region_h = self.width + extx, self.height + exty
        downscale_ratio = max(min(region_w / fg_w, region_h / fg_h), 0.8)
        if downscale_ratio < 1:
            fg_h = int(downscale_ratio * fg_h)
            fg_w = int(downscale_ratio * fg_w)
        
        max_x, max_y = self.bright + extx - fg_w, self.bbottom + exty - fg_h
        max_x = min(im_w+extxb-fg_w, max_x)
        max_y = min(im_h+extyb-fg_h, max_y)
        min_x = max(min(self.bright + extx - fg_w, self.bleft - extx), -extx)
        min_x = max(-extxb, min_x)
        min_y = max(min(self.bbottom + exty - fg_h, self.btop - exty), -exty)
        min_y = max(-extyb, min_y)
        px, py = min_x, min_y
        if min_x < max_x:
            if random_sample:
                px = random.randint(min_x, max_x)
            else:
                px = int((min_x + max_x) / 2)
        if min_y < max_y:
            if random_sample:
                py = random.randint(min_y, max_y)
            else:
                py = int((min_y + max_y) / 2)
        return px, py, downscale_ratio

    def drawpartition(self, image: np.ndarray, color = None):
        if color is None:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if not self.is_leaf():
            cv2.rectangle(image, (self.bleft, self.btop), (self.bright, self.bbottom), color, 2)
        if not self.is_leaf():
            c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            self.left.drawpartition(image, c)
            self.right.drawpartition(image, c)
            self.top.drawpartition(image, c)
            self.bottom.drawpartition(image, c)


def paste_one_fg(fg_pil: Image, bg: Image, segments: np.ndarray, px: int, py: int, seg_color: Tuple, cal_area=True):
    
    fg_h, fg_w = fg_pil.height, fg_pil.width
    im_h, im_w = bg.height, bg.width
    
    bg.paste(fg_pil, (px, py), mask=fg_pil)
    

    bgx1, bgx2, bgy1, bgy2 = px, px+fg_w, py, py+fg_h
    fgx1, fgx2, fgy1, fgy2 = 0, fg_w, 0, fg_h
    if bgx1 < 0:
        fgx1 = -bgx1
        bgx1 = 0
    if bgy1 < 0:
        fgy1 = -bgy1
        bgy1 = 0
    if bgx2 > im_w:
        fgx2 = im_w - bgx2
        bgx2 = im_w
    if bgy2 > im_h:
        fgy2 = im_h - bgy2
        bgy2 = im_h

    fg_mask = np.array(fg_pil)[fgy1: fgy2, fgx1: fgx2, 3] > 30
    segments[bgy1: bgy2, bgx1: bgx2][np.where(fg_mask)] = seg_color

    if cal_area:
        area = fg_mask.sum()
    else:
        area = 1
    bbox = [bgx1, bgy1, bgx2-bgx1, bgy2-bgy1]
    return area, bbox, [bgx1, bgy1, bgx2, bgy2]


def partition_paste(fg_list, bg: Image):
    segments_info = []
    
    fg_list.sort(key = lambda x: x['image'].shape[0] * x['image'].shape[1], reverse=True)
    pnode: PartitionTree = None
    im_h, im_w = bg.height, bg.width
    
    ptree = PartitionTree(0, 0, bg.width, bg.height)
    
    segments = np.zeros((im_h, im_w, 3), np.uint8)
    for ii, fg_dict in enumerate(fg_list):
        fg = fg_dict['image']
        fg_h, fg_w = fg.shape[:2]
        pnode, _ = ptree.prefer_partition(fg_h, fg_w)
        px, py, downscale_ratio = pnode.new_random_pos(fg_h, fg_w, im_h, im_w, True)
        
        fg_pil = Image.fromarray(fg)
        if downscale_ratio < 1:
            fg_pil = fg_pil.resize((int(fg_w * downscale_ratio), int(fg_h * downscale_ratio)), resample=Image.Resampling.LANCZOS)
            # fg_h, fg_w = fg_pil.height, fg_pil.width

        seg_color = COLOR_PALETTE[ii]
        area, bbox, xyxy = paste_one_fg(fg_pil, bg, segments, px,py, seg_color, cal_area=False)
        pnode.new_partition(xyxy)

        segments_info.append({
            'id': rgb2id(seg_color),
            'bbox': bbox,
            'area': area
        })
        
    return segments_info, segments
        # if downscale_ratio < 1:
        #     fg_pil = fg_pil.resize((int(fg_w * downscale_ratio), int(fg_h * downscale_ratio)), resample=Image.Resampling.LANCZOS)
        #     fg_h, fg_w = fg_pil.height, fg_pil.width


def gen_fg_regbboxes(fg_list: List[Dict], tgt_size: int, min_overlap=0.15, max_overlap=0.8):
    
    def _sample_y(h):
        y = (tgt_size - h) // 2
        if y > 0:
            yrange = min(y, h // 4)
            y += random.randint(-yrange, yrange)
            return y
        else:
            return 0

    shape_list = []
    depth_list = []
    

    for fg_dict in fg_list:
        shape_list.append(fg_dict['image'].shape[:2])
    
    shape_list = np.array(shape_list)
    depth_list = np.random.random(len(fg_list))
    depth_list[shape_list[..., 1] > 0.6 * tgt_size] += 1 

    # num_fg = len(fg_list)
    # grid_sample = random.random() < 0.4 or num_fg > 6
    # grid_sample = grid_sample and num_fg < 9 and num_fg > 3
    # grid_sample = False
    # if grid_sample:
    #     grid_pos = np.arange(9)
    #     np.random.shuffle(grid_pos)
    #     grid_pos = grid_pos[: num_fg]
    #     grid_x = grid_pos % 3
    #     grid_y = grid_pos // 3

    # else:
    pos_list = [[0, _sample_y(shape_list[0][0])]]
    pre_overlap = 0
    for ii, ((h, w), d) in enumerate(zip(shape_list[1:], depth_list[1:])):
        (preh, prew), predepth, (prex, prey) = shape_list[ii], depth_list[ii], pos_list[ii]
        
        isfg = d < predepth
        y = _sample_y(h)
        x = prex+prew
        if isfg:
            min_x = max_x = x
            if pre_overlap < max_overlap:
                min_x -= (max_overlap - pre_overlap) * prew
                min_x = int(min_x)
                if pre_overlap < min_overlap:
                    max_x -= (min_overlap - pre_overlap) * prew
                    max_x = int(max_x)
                x = random.randint(min_x, max_x)
            pre_overlap = 0
        else:
            overlap = random.uniform(min_overlap, max_overlap)
            x -= int(overlap * w)
            area = h * w
            overlap_area = bbox_overlap_area([x, y, w, h], [prex, prey, prew, preh])
            pre_overlap = overlap_area / area

        pos_list.append([x, y])

    pos_list = np.array(pos_list)
    last_x2 = pos_list[-1][0] + shape_list[-1][1]
    valid_shiftx = tgt_size - last_x2
    if valid_shiftx > 0:
        shiftx = random.randint(0, valid_shiftx)
        pos_list[:, 0] += shiftx
    else:
        pos_list[:, 0] += valid_shiftx // 2

    for pos, fg_dict, depth in zip(pos_list, fg_list, depth_list):
        fg_dict['pos'] = pos
        fg_dict['depth'] = depth
    fg_list.sort(key=lambda x: x['depth'], reverse=True)



def regular_paste(fg_list, bg: Image, regen_bboxes=False):
    segments_info = []
    im_h, im_w = bg.height, bg.width
    
    if regen_bboxes:
        random.shuffle(fg_list)
        gen_fg_regbboxes(fg_list, im_h)    
    
    segments = np.zeros((im_h, im_w, 3), np.uint8)
    for ii, fg_dict in enumerate(fg_list):
        fg = fg_dict['image']
        
        px, py = fg_dict.pop('pos')
        fg_pil = Image.fromarray(fg)

        seg_color = COLOR_PALETTE[ii]
        area, bbox, xyxy = paste_one_fg(fg_pil, bg, segments, px,py, seg_color, cal_area=True)

        segments_info.append({
            'id': rgb2id(seg_color),
            'bbox': bbox,
            'area': area
        })
        
    return segments_info, segments