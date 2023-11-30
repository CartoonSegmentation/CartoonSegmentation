import numpy as np
from typing import List, Union, Tuple, Dict
import random
from PIL import Image
import cv2
import imageio, os
import os.path as osp
from tqdm import tqdm
from panopticapi.utils import rgb2id 
import traceback

from utils.io_utils import mask2rle, dict2json, fgbg_hist_matching
from utils.logger import LOGGER
from utils.constants import CATEGORIES, IMAGE_ID_ZFILL
from .transforms import get_fg_transforms, get_bg_transforms, quantize_image, resize2height, rotate_image
from .sampler import random_load_valid_bg, random_load_valid_fg, NameSampler, NormalSampler, PossionSampler, PersonBBoxSampler
from .paste_methods import regular_paste, partition_paste


def syn_animecoco_dataset(
    bg_list: List, fg_info_list: List[Dict], dataset_save_dir: str, policy: str='train', 
    tgt_size=640, syn_num_multiplier=2.5, regular_paste_prob=0.4, person_paste_prob=0.4,
    max_syn_num=-1, image_id_start=0, obj_id_start=0, hist_match_prob=0.2, quantize_prob=0.25):

    LOGGER.info(f'syn data policy: {policy}')
    LOGGER.info(f'background: {len(bg_list)} foreground: {len(fg_info_list)}')

    numfg_sampler = PossionSampler(min_val=1, max_val=9, lam=2.5)
    numfg_regpaste_sampler = PossionSampler(min_val=2, max_val=9, lam=3.5)
    regpaste_size_sampler = NormalSampler(scalar=tgt_size, to_int=True, max_scale=0.75)
    color_correction_sampler = NameSampler({'hist_match': hist_match_prob, 'quantize': quantize_prob}, )
    paste_method_sampler = NameSampler({'regular': regular_paste_prob, 'personbbox': person_paste_prob, 
                            'partition': 1-regular_paste_prob-person_paste_prob})

    fg_transform = get_fg_transforms(tgt_size, transform_variant=policy)
    fg_distort_transform = get_fg_transforms(tgt_size, transform_variant='distort_only')
    bg_transform = get_bg_transforms('train', tgt_size)

    image_id = image_id_start + 1
    obj_id = obj_id_start + 1

    det_annotations, image_meta = [], []

    syn_num = int(syn_num_multiplier * len(fg_info_list))
    if max_syn_num > 0:
        syn_num = max_syn_num

    ann_save_dir = osp.join(dataset_save_dir, 'annotations')
    image_save_dir = osp.join(dataset_save_dir, policy)

    if not osp.exists(image_save_dir):
        os.makedirs(image_save_dir)
    if not osp.exists(ann_save_dir):
        os.makedirs(ann_save_dir)

    is_train =  policy == 'train'
    if is_train:
        jpg_save_quality = [75, 85, 95]
    else:
        jpg_save_quality = [95]

    if isinstance(fg_info_list[0], str):
        for ii, fgp in enumerate(fg_info_list):
            if isinstance(fgp, str):
                fg_info_list[ii] = {'file_path': fgp, 'tag_string': [], 'danbooru': False, 'category_id': 0}

    if person_paste_prob > 0:
        personbbox_sampler = PersonBBoxSampler(
            'data/cocoperson_bbox_samples.json', fg_info_list, 
            fg_transform=fg_distort_transform if is_train else None, is_train=is_train)

    total = tqdm(range(syn_num))
    for fin in total:
        try:
            paste_method = paste_method_sampler.sample()

            fgs = []
            if paste_method == 'regular':
                num_fg = numfg_regpaste_sampler.sample()
                size = regpaste_size_sampler.sample()
                while len(fgs) < num_fg:
                    tgt_height = int(random.uniform(0.7, 1.2) * size)
                    fg, fginfo = random_load_valid_fg(fg_info_list)
                    fg = resize2height(fg, tgt_height)
                    if is_train:
                        fg = fg_distort_transform(image=fg)['image']
                        rotate_deg = random.randint(-40, 40)
                    else:
                        rotate_deg = random.randint(-30, 30)
                    if random.random() < 0.3:
                        fg = rotate_image(fg, rotate_deg, alpha_crop=True)
                    fgs.append({'image': fg, 'fginfo': fginfo})
                    while len(fgs) < num_fg and random.random() < 0.15:
                        fgs.append({'image': fg, 'fginfo': fginfo})
            elif paste_method == 'personbbox':
                fgs = personbbox_sampler.sample_matchfg(tgt_size)
            else:
                num_fg = numfg_sampler.sample()
                fgs = []
                for ii in range(num_fg):
                    fg, fginfo = random_load_valid_fg(fg_info_list)
                    fg = fg_transform(image=fg)['image']
                    h, w = fg.shape[:2]
                    if num_fg > 6:
                        downscale = min(tgt_size / 2.5 / w, tgt_size / 2.5 / h)
                        if downscale < 1:
                            fg = cv2.resize(fg, (int(w * downscale), int(h * downscale)), interpolation=cv2.INTER_AREA)
                    fgs.append({'image': fg, 'fginfo': fginfo})

            bg = random_load_valid_bg(bg_list)
            bg = bg_transform(image=bg)['image']

            color_correct = color_correction_sampler.sample()

            if color_correct == 'hist_match':
                fgbg_hist_matching(fgs, bg)
            
            bg: Image = Image.fromarray(bg)

            if paste_method == 'regular':
                segments_info, segments = regular_paste(fgs, bg, regen_bboxes=True) 
            elif paste_method == 'personbbox':
                segments_info, segments = regular_paste(fgs, bg, regen_bboxes=False) 
            elif paste_method == 'partition':
                segments_info, segments = partition_paste(fgs, bg, )
            else:
                print(f'invalid paste method: {paste_method}')
                raise NotImplementedError 

            image = np.array(bg)
            if color_correct == 'quantize':
                mask = cv2.inRange(segments, np.array([0,0,0]), np.array([0,0,0]))
                # cv2.imshow("mask", mask)
                image = quantize_image(image, random.choice([12, 16, 32]), 'kmeans', mask=mask)[0]

            # postprocess & check if instance is valid
            for ii, segi in enumerate(segments_info):
                if segi['area'] == 0:
                    continue
                x, y, w, h = segi['bbox']
                x2, y2 = x+w, y+h
                c = segments[y: y2, x: x2]
                pan_png = rgb2id(c)
                cmask = (pan_png == segi['id'])
                area = cmask.sum()
                
                if paste_method != 'partition' and \
                    area / (fgs[ii]['image'][..., 3] > 30).sum() < 0.25:
                    # cv2.imshow('im', fgs[ii]['image'])
                    # cv2.imshow('mask', fgs[ii]['image'][..., 3])
                    # cv2.imshow('seg', segments)
                    # cv2.waitKey(0)
                    cmask_ids = np.where(cmask)
                    segments[y: y2, x: x2][cmask_ids] = 0
                    image[y: y2, x: x2][cmask_ids] = (127, 127, 127)
                    continue
                
                cmask = cmask.astype(np.uint8) * 255
                dx, dy, w, h = cv2.boundingRect(cv2.findNonZero(cmask))
                _bbox = [dx + x, dy + y, w, h]

                seg = cv2.copyMakeBorder(cmask, y, tgt_size-y2, x, tgt_size-x2, cv2.BORDER_CONSTANT) > 0
                assert seg.shape[0] == tgt_size and seg.shape[1] == tgt_size
                segmentation = mask2rle(seg)

                det_annotations.append({
                    'id': obj_id,
                    'category_id': fgs[ii]['fginfo']['category_id'],
                    'iscrowd': 0,
                    'segmentation': segmentation,
                    'image_id': image_id,
                    'area': area,
                    'tag_string': fgs[ii]['fginfo']['tag_string'],
                    'tag_string_character': fgs[ii]['fginfo']['tag_string_character'],
                    'bbox': [float(c) for c in _bbox]
                })

                obj_id += 1
                # cv2.imshow('c', cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
                # cv2.imshow('cmask', cmask)
                # cv2.waitKey(0)

            image_id_str = str(image_id).zfill(IMAGE_ID_ZFILL)
            image_file_name = image_id_str + '.jpg'
            image_meta.append({
                "id": image_id,"height": tgt_size,"width": tgt_size, "file_name": image_file_name, "id": image_id
            })

            # LOGGER.info(f'paste method: {paste_method} color correct: {color_correct}')
            # cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imshow('segments', cv2.cvtColor(segments, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)

            imageio.imwrite(osp.join(image_save_dir, image_file_name), image, quality=random.choice(jpg_save_quality))
            image_id += 1

        except:
            LOGGER.error(traceback.format_exc())
            continue

    det_meta = {
        "info": {},
        "licenses": [],
        "images": image_meta,
        "annotations": det_annotations,
        "categories": CATEGORIES
    }

    detp = osp.join(ann_save_dir, f'det_{policy}.json')
    dict2json(det_meta, detp)
    LOGGER.info(f'annotations saved to {detp}')

    return image_id, obj_id