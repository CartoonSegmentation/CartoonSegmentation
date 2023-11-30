import numpy as np
from random import choice as rchoice
from random import randint
import random
import cv2, traceback, imageio
import os.path as osp

from typing import Optional, List, Union, Tuple, Dict
from utils.io_utils import imread_nogrey_rgb, json2dict
from .transforms import rotate_image
from utils.logger import LOGGER


class NameSampler:

    def __init__(self, name_prob_dict, sample_num=2048) -> None:
        self.name_prob_dict = name_prob_dict
        self._id2name = list(name_prob_dict.keys())
        self.sample_ids = []

        total_prob = 0.
        for ii, (_, prob) in enumerate(name_prob_dict.items()):
            tgt_num = int(prob * sample_num)
            total_prob += prob
            if tgt_num > 0:
                self.sample_ids += [ii] * tgt_num

        nsamples = len(self.sample_ids)
        assert prob <= 1
        if prob < 1 and nsamples < sample_num:
            self.sample_ids += [len(self._id2name)] * (sample_num - nsamples)
            self._id2name.append('_')

    def sample(self) -> str:
        return self._id2name[rchoice(self.sample_ids)]


class PossionSampler:
    def __init__(self, lam=3, min_val=1, max_val=8) -> None:
        self._distr = np.random.poisson(lam, 1024)
        invalid = np.where(np.logical_or(self._distr<min_val, self._distr > max_val))
        self._distr[invalid] = np.random.randint(min_val, max_val, len(invalid[0]))

    def sample(self) -> int:
        return rchoice(self._distr)


class NormalSampler:
    def __init__(self, loc=0.33, std=0.2, min_scale=0.15, max_scale=0.85, scalar=1, to_int = True):
        s = np.random.normal(loc, std, 4096)
        valid = np.where(np.logical_and(s>min_scale, s<max_scale))
        self._distr = s[valid] * scalar
        if to_int:
            self._distr = self._distr.astype(np.int32)

    def sample(self) -> int:
        return rchoice(self._distr)


class PersonBBoxSampler:

    def __init__(self, sample_path: Union[str, List]='data/cocoperson_bbox_samples.json', fg_info_list: List = None, fg_transform=None, is_train=True) -> None:
        if isinstance(sample_path, str):
            sample_path = [sample_path]
        self.bbox_list = []
        for sp in sample_path:
            bboxlist = json2dict(sp)
            for bboxes in bboxlist:
                if isinstance(bboxes, dict):
                    bboxes = bboxes['bboxes']
                bboxes = np.array(bboxes)
                bboxes[:, [0, 1]] -= bboxes[:, [0, 1]].min(axis=0)
                self.bbox_list.append(bboxes)

        self.fg_info_list = fg_info_list
        self.fg_transform = fg_transform
        self.is_train = is_train

    def sample(self, tgt_size: int, scale_range=(1, 1), size_thres=(0.02, 0.85)) -> List[np.ndarray]:
        bboxes_normalized = rchoice(self.bbox_list)
        if scale_range[0] != 1 or scale_range[1] != 1:
            bbox_scale = random.uniform(scale_range[0], scale_range[1])
        else:
            bbox_scale = 1
        bboxes = (bboxes_normalized * tgt_size * bbox_scale).astype(np.int32)
        
        xyxy_array = np.copy(bboxes)
        xyxy_array[:, [2, 3]] += xyxy_array[:, [0, 1]]
        x_max, y_max = xyxy_array[:, 2].max(), xyxy_array[:, 3].max()

        x_shift = tgt_size - x_max
        x_shift = randint(0, x_shift) if x_shift > 0 else 0
        y_shift = tgt_size - y_max
        y_shift = randint(0, y_shift) if y_shift > 0 else 0
        
        bboxes[:, [0, 1]] += [x_shift, y_shift]
        valid_bboxes = []
        max_size = size_thres[1] * tgt_size
        min_size = size_thres[0] * tgt_size
        for bbox in bboxes:
            w = min(bbox[2], tgt_size - bbox[0])
            h = min(bbox[3], tgt_size - bbox[1])
            if max(h, w) < max_size and min(h, w) > min_size:
                valid_bboxes.append(bbox)
        return valid_bboxes

    def sample_matchfg(self, tgt_size: int):
        while True:
            bboxes = self.sample(tgt_size, (1.1, 1.8))
            if len(bboxes) > 0:
                break
        MIN_FG_SIZE = 20
        num_fg = len(bboxes)
        rotate = 20 if self.is_train else 15
        fgs = random_load_nfg(num_fg, self.fg_info_list, random_rotate_prob=0.33, random_rotate=rotate)
        assert len(fgs) == num_fg

        bboxes.sort(key=lambda x: x[2] / x[3])
        fgs.sort(key=lambda x: x['asp_ratio'])

        for fg, bbox in zip(fgs, bboxes):
            x, y, w, h = bbox
            img = fg['image']
            im_h, im_w = img.shape[:2]
            if im_h < h and im_w < w:
                scale = min(h / im_h, w / im_w)
                new_h, new_w = int(scale * im_h), int(scale * im_w)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scale_h, scale_w = min(1, h / im_h), min(1, w / im_w)
                scale = (scale_h + scale_w) / 2
                if scale < 1:
                    new_h, new_w = max(int(scale * im_h), MIN_FG_SIZE), max(int(scale * im_w), MIN_FG_SIZE)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if self.fg_transform is not None:
                img = self.fg_transform(image=img)['image']

            im_h, im_w = img.shape[:2]
            fg['image'] = img
            px = int(x + w / 2 - im_w / 2)
            py = int(y + h / 2 - im_h / 2)
            fg['pos'] = (px, py)

        random.shuffle(fgs)

        slist, llist = [], []
        large_size = int(tgt_size * 0.55)
        for fg in fgs:
            if max(fg['image'].shape[:2]) > large_size:
                llist.append(fg)
            else:
                slist.append(fg)
        return llist + slist


def random_load_nfg(num_fg: int, fg_info_list: List[Union[Dict, str]], random_rotate=0, random_rotate_prob=0.):
    fgs = []
    while len(fgs) < num_fg:
        fg, fginfo = random_load_valid_fg(fg_info_list)
        if random.random() < random_rotate_prob:
            rotate_deg = randint(-random_rotate, random_rotate)
            fg = rotate_image(fg, rotate_deg, alpha_crop=True)
  
        asp_ratio = fg.shape[1] / fg.shape[0]
        fgs.append({'image': fg, 'asp_ratio': asp_ratio, 'fginfo': fginfo})
        while len(fgs) < num_fg and random.random() < 0.12:
            fgs.append({'image': fg, 'asp_ratio': asp_ratio, 'fginfo': fginfo})
    
    return fgs


def random_load_valid_fg(fg_info_list: List[Union[Dict, str]]) -> Tuple[np.ndarray, Dict]:
    while True:
        item = fginfo = rchoice(fg_info_list)

        file_path = fginfo['file_path']
        if 'root_dir' in fginfo and fginfo['root_dir']:
            file_path = osp.join(fginfo['root_dir'], file_path)
        
        try:
            fg = imageio.imread(file_path)
        except:
            LOGGER.error(traceback.format_exc())
            LOGGER.error(f'invalid fg: {file_path}')
            fg_info_list.remove(item)
            continue

        c = 1
        if len(fg.shape) == 3:
            c = fg.shape[-1]
        if c != 4:
            LOGGER.warning(f'fg {file_path} doesnt have alpha channel')
            fg_info_list.remove(item)
        else:
            if 'xyxy' in fginfo:
                x1, y1, x2, y2 = fginfo['xyxy']
            else:
                oh, ow = fg.shape[:2]
                ksize = 5
                mask = cv2.blur(fg[..., 3], (ksize,ksize))
                _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
                
                x1, y1, w, h = cv2.boundingRect(cv2.findNonZero(mask))
                x2, y2 = x1 + w, y1 + h
                if oh - h > 15 or ow - w > 15:
                    crop = True
                else:
                    x1 = y1 = 0
                    x2, y2 = ow, oh
                    
            fginfo['xyxy'] = [x1, y1, x2, y2]
            fg = fg[y1: y2, x1: x2]
            return fg, fginfo


def random_load_valid_bg(bg_list: List[str]) -> np.ndarray:
    while True:
        try:
            bgp = rchoice(bg_list)
            return imread_nogrey_rgb(bgp)
        except:
            LOGGER.error(traceback.format_exc())
            LOGGER.error(f'invalid bg: {bgp}')
            bg_list.remove(bgp)
            continue