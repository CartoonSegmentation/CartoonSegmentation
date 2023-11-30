import albumentations as A

from torch.utils.data import Dataset, DataLoader
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
import random
import os.path as osp
import cv2
import numpy as np
from scipy.ndimage import distance_transform_bf, distance_transform_edt, distance_transform_cdt


def is_grey(img: np.ndarray):
    if len(img.shape) == 3 and img.shape[2] == 3:
        return False
    else:
        return True


def square_pad_resize(img: np.ndarray, tgt_size: int, pad_value = (0, 0, 0)):
    h, w = img.shape[:2]
    pad_h, pad_w = 0, 0
    
    # make square image
    if w < h:
        pad_w = h - w
        w += pad_w
    elif h < w:
        pad_h = w - h
        h += pad_h

    pad_size = tgt_size - h
    if pad_size > 0:
        pad_h += pad_size
        pad_w += pad_size

    if pad_h > 0 or pad_w > 0:    
        c = 1
        if is_grey(img):
            if isinstance(pad_value, tuple):
                pad_value = pad_value[0]
        else:
            if isinstance(pad_value, int):
                pad_value = (pad_value, pad_value, pad_value)

        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=pad_value)

    resize_ratio = tgt_size / img.shape[0]
    if resize_ratio < 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_AREA)
    elif resize_ratio > 1:
        img = cv2.resize(img, (tgt_size, tgt_size), interpolation=cv2.INTER_LINEAR)
        
    return img, resize_ratio, pad_h, pad_w


class MaskRefineDataset(Dataset):

    def __init__(self, 
                 refine_ann_path: str, 
                 data_root: str, 
                 load_instance_mask: bool = True, 
                 aug_ins_prob: float = 0.,
                 ins_rect_prob: float = 0.,
                 output_size: int = 720,
                 augmentation: bool = False,
                 with_distance: bool = False):
        self.load_instance_mask = load_instance_mask
        self.ann_util = COCO(refine_ann_path)
        self.img_ids = self.ann_util.getImgIds()
        self.set_load_method(load_instance_mask)
        self.data_root = data_root

        self.ins_rect_prob = ins_rect_prob
        self.aug_ins_prob = aug_ins_prob
        self.augmentation = augmentation
        if augmentation:
            transform = [
                A.OpticalDistortion(),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.Posterize(),
                A.CropAndPad(percent=0.1, p=0.3, pad_mode=cv2.BORDER_CONSTANT, pad_cval=0, pad_cval_mask=0, keep_size=True),
                A.RandomContrast(), 
                A.Rotate(30, p=0.3, mask_value=0, border_mode=cv2.BORDER_CONSTANT)
            ]
            self._aug_transform = A.Compose(transform)
        else:
            self._aug_transform = None

        self.output_size = output_size
        self.with_distance = with_distance

    def set_output_size(self, size: int):
        self.output_size = size

    def set_load_method(self, load_instance_mask: bool):
        if load_instance_mask:
            self._load_mask = self._load_with_instance
        else:
            self._load_mask = self._load_without_instance

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_meta = self.ann_util.imgs[img_id]
        img_path = osp.join(self.data_root, img_meta['file_name'])
        img = cv2.imread(img_path)

        annids = self.ann_util.getAnnIds([img_id])
        if len(annids) > 0:
            ann = random.choice(annids)
            ann = self.ann_util.anns[ann]
            assert ann['image_id'] == img_id
        else:
            ann = None
        
        return self._load_mask(img, ann)
    
    def transform(self, img: np.ndarray, mask: np.ndarray, ins_seg: np.ndarray = None) -> dict:
        if ins_seg is not None:
            use_seg = True
        else:
            use_seg = False

        if self.augmentation:
            masks = [mask]
            if use_seg:
                masks.append(ins_seg)
            data = self._aug_transform(image=img, masks=masks)
            img = data['image']
            masks = data['masks']
            mask = masks[0]
            if use_seg:
                ins_seg = masks[1]

        img = square_pad_resize(img, self.output_size, random.randint(0, 255))[0]
        mask = square_pad_resize(mask, self.output_size, 0)[0]
        if ins_seg is not None:
            ins_seg = square_pad_resize(ins_seg, self.output_size, 0)[0]

        img = (img.astype(np.float32) / 255.).transpose((2, 0, 1))
        mask = mask[None, ...]


        if use_seg:
            ins_seg = ins_seg[None, ...]
            img = np.concatenate((img, ins_seg), axis=0)

        data = {'img': img, 'mask': mask}
        if self.with_distance:
            dist = distance_transform_edt(mask[0])
            dist_max = dist.max()
            if dist_max != 0:
                dist = 1 - dist / dist_max
                # diff_mat = cv2.bitwise_xor(mask[0], ins_seg[0])
                # dist = dist + diff_mat + 0.2
                dist = dist + 0.2
                dist = dist.size / (dist.sum() + 1) * dist
                dist = np.clip(dist, 0, 20)
            else:
                dist = np.ones_like(dist)
                # print(dist.max(), dist.min())
            data['dist_weight'] = dist[None, ...]
        return data

    def _load_with_instance(self, img: np.ndarray, ann: dict):
        if ann is None:
            mask = np.zeros(img.shape[:2], dtype=np.float32)
            ins_seg = mask
        else:
            mask = maskUtils.decode(ann['segmentation']).astype(np.float32)
            if self.augmentation and random.random() < self.ins_rect_prob:
                ins_seg = np.zeros_like(mask)
                bbox = [int(b) for b in ann['bbox']]
                ins_seg[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]] = 1
            elif len(ann['pred_segmentations']) > 0:
                ins_seg = random.choice(ann['pred_segmentations'])
                ins_seg = maskUtils.decode(ins_seg).astype(np.float32)
            else:
                ins_seg = mask
            if self.augmentation and random.random() < self.aug_ins_prob:
                ksize = random.choice([1, 3, 5, 7])
                ksize = ksize * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(ksize, ksize))
                if random.random() < 0.5:
                    ins_seg = cv2.dilate(ins_seg, kernel)
                else:
                    ins_seg = cv2.erode(ins_seg, kernel)

        return self.transform(img, mask, ins_seg)

    def _load_without_instance(self, img: np.ndarray, ann: dict):
        if ann is None:
            mask = np.zeros(img.shape[:2], dtype=np.float32)
        else:
            mask = maskUtils.decode(ann['segmentation']).astype(np.float32)
        return self.transform(img, mask)

    def __len__(self):
        return len(self.img_ids)


if __name__ == '__main__':
    ann_path = r'workspace/test_syndata/annotations/refine_train.json'
    data_root = r'workspace/test_syndata/train'

    ann_path = r'workspace/test_syndata/annotations/refine_train.json'
    data_root = r'workspace/test_syndata/train'
    aug_ins_prob = 0.5
    load_instance_mask = True
    ins_rect_prob = 0.25
    output_size = 640
    augmentation = True

    random.seed(0)

    md = MaskRefineDataset(ann_path, data_root, load_instance_mask, aug_ins_prob, ins_rect_prob, output_size, augmentation, with_distance=True)
    
    dl = DataLoader(md, batch_size=1, shuffle=False, persistent_workers=True,
                                  num_workers=1, pin_memory=True)
    for data in dl:
        img = data['img'].cpu().numpy()
        img = (img[0, :3].transpose((1, 2, 0)) * 255).astype(np.uint8)
        mask = (data['mask'].cpu().numpy()[0][0] * 255).astype(np.uint8)
        if load_instance_mask:
            ins = (data['img'].cpu().numpy()[0][3] * 255).astype(np.uint8)
            cv2.imshow('ins', ins)
        dist = data['dist_weight'].cpu().numpy()[0][0]
        dist = (dist / dist.max() * 255).astype(np.uint8)
        cv2.imshow('img', img)
        cv2.imshow('mask', mask)
        cv2.imshow('dist_weight', dist)
        cv2.waitKey(0)

        # cv2.imwrite('')