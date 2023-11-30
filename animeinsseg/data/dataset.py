import os.path as osp
import numpy as np
from typing import List, Optional, Sequence, Tuple, Union
import copy
from time import time
import mmcv
from mmcv.transforms import to_tensor
from mmdet.datasets.transforms import LoadAnnotations, RandomCrop, PackDetInputs, Mosaic, CachedMosaic, CachedMixUp, FilterAnnotations
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS, TRANSFORMS
from numpy import random
from mmdet.structures.bbox import autocast_box_type, BaseBoxes
from mmengine.structures import InstanceData, PixelData
from mmdet.structures import DetDataSample
from utils.io_utils import bbox_overlap_xy
from utils.logger import LOGGER

@DATASETS.register_module()
class AnimeMangaMixedDataset(CocoDataset):

    def __init__(self, animeins_root: str = None, animeins_annfile: str = None, manga109_annfile: str = None, manga109_root: str = None, *args, **kwargs) -> None:
        self.animeins_annfile = animeins_annfile
        self.animeins_root = animeins_root
        self.manga109_annfile = manga109_annfile
        self.manga109_root = manga109_root
        self.cat_ids = []
        self.cat_img_map = {}
        super().__init__(*args, **kwargs)
        LOGGER.info(f'total num data: {len(self.data_list)}')


    def parse_data_info(self, raw_data_info: dict, data_prefix: str) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(data_prefix, img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info


    def load_data_list(self) -> List[dict]:
        data_lst = []
        if self.manga109_root is not None:
            data_lst += self._data_list(self.manga109_annfile, osp.join(self.manga109_root, 'images'))
            # if len(data_lst) > 8000:
            #     data_lst = data_lst[:500]
            LOGGER.info(f'num data from manga109: {len(data_lst)}')
        if self.animeins_root is not None:
            animeins_annfile = osp.join(self.animeins_root, self.animeins_annfile)
            data_prefix = osp.join(self.animeins_root, self.data_prefix['img'])
            anime_lst = self._data_list(animeins_annfile, data_prefix)
            # if len(anime_lst) > 8000:
            #     anime_lst = anime_lst[:500]
            data_lst += anime_lst
            LOGGER.info(f'num data from animeins: {len(data_lst)}')
        return data_lst

    def _data_list(self, annfile: str, data_prefix: str) -> List[dict]:
        """Load annotations from an annotation file named as ``ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with self.file_client.get_local_path(annfile) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        for key, val in cat_img_map.items():
            if key in self.cat_img_map:
                self.cat_img_map[key] += val
            else:
                self.cat_img_map[key] = val

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            }, data_prefix)
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{annfile}' are not unique!"

        del self.coco

        return data_list



@TRANSFORMS.register_module()
class LoadAnnotationsNoSegs(LoadAnnotations):

    def _process_masks(self, results: dict) -> list:
        """Process gt_masks and filter invalid polygons.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            list: Processed gt_masks.
        """
        gt_masks = []
        gt_ignore_flags = []
        gt_ignore_mask_flags = []
        for instance in results.get('instances', []):
            gt_mask = instance['mask']
            ignore_mask = False
            # If the annotation of segmentation mask is invalid,
            # ignore the whole instance.
            if isinstance(gt_mask, list):
                gt_mask = [
                    np.array(polygon) for polygon in gt_mask
                    if len(polygon) % 2 == 0 and len(polygon) >= 6
                ]
                if len(gt_mask) == 0:
                    # ignore this instance and set gt_mask to a fake mask
                    instance['ignore_flag'] = 1
                    gt_mask = [np.zeros(6)]
            elif not self.poly2mask:
                # `PolygonMasks` requires a ploygon of format List[np.array],
                # other formats are invalid.
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and \
                    not (gt_mask.get('counts') is not None and
                         gt_mask.get('size') is not None and
                         isinstance(gt_mask['counts'], (list, str))):
                # if gt_mask is a dict, it should include `counts` and `size`,
                # so that `BitmapMasks` can uncompressed RLE
                # instance['ignore_flag'] = 1
                ignore_mask = True
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            # re-process gt_ignore_flags
            gt_ignore_flags.append(instance['ignore_flag'])
            gt_ignore_mask_flags.append(ignore_mask)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        results['gt_ignore_mask_flags'] = np.array(gt_ignore_mask_flags, dtype=bool)
        return gt_masks

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)
        if self.poly2mask:
            p2masks = []
            if len(gt_masks) > 0:
                for ins, mask, ignore_mask in zip(results['instances'], gt_masks, results['gt_ignore_mask_flags']):
                    bbox = [int(c) for c in ins['bbox']]
                    if ignore_mask:
                        m = np.zeros((h, w), dtype=np.uint8)
                        m[bbox[1]:bbox[3], bbox[0]: bbox[2]] = 255
                        # m[bbox[1]:bbox[3], bbox[0]: bbox[2]]
                        p2masks.append(m)
                    else:
                        p2masks.append(self._poly2mask(mask, h, w))
                # import cv2
                # # cv2.imwrite('tmp_mask.png', p2masks[-1] * 255)
                # cv2.imwrite('tmp_img.png', results['img'])
                # cv2.imwrite('tmp_bbox.png', m * 225)
                # print(p2masks[-1].shape, p2masks[-1].dtype)
            gt_masks = BitmapMasks(p2masks, h, w)
        else:
            # fake polygon masks will be ignored in `PackDetInputs`
            gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)

        return results
        


@TRANSFORMS.register_module()
class PackDetIputsNoSeg(PackDetInputs):

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_ignore_mask_flags': 'ignore_mask',
        'gt_masks': 'masks'
    }

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results



def translate_bitmapmask(bitmap_masks: BitmapMasks,
              out_shape,
              offset_x,
              offset_y,):
              
    if len(bitmap_masks.masks) == 0:
        translated_masks = np.empty((0, *out_shape), dtype=np.uint8)
    else:
        masks = bitmap_masks.masks
        out_h, out_w = out_shape
        mask_h, mask_w = masks.shape[1:]

        translated_masks = np.zeros((masks.shape[0], *out_shape),
                                dtype=masks.dtype)

        ix, iy = bbox_overlap_xy([0, 0, out_w, out_h], [offset_x, offset_y, mask_w, mask_h])
        if ix > 2 and iy > 2:
            if offset_x > 0:
                mx1 = 0
                tx1 = offset_x
            else:
                mx1 = -offset_x
                tx1 = 0
            mx2 = min(out_w - offset_x, mask_w)
            tx2 = tx1 + mx2 - mx1

            if offset_y > 0:
                my1 = 0
                ty1 = offset_y
            else:
                my1 = -offset_y
                ty1 = 0
            my2 = min(out_h - offset_y, mask_h)
            ty2 = ty1 + my2 - my1

            translated_masks[:, ty1: ty2, tx1: tx2] = \
                masks[:, my1: my2, mx1: mx2]

    return BitmapMasks(translated_masks, *out_shape)


@TRANSFORMS.register_module()
class CachedMosaicNoSeg(CachedMosaic):

    @autocast_box_type()
    def transform(self, results: dict) -> dict:

        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        self.results_cache.append(copy.deepcopy(results))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 4:
            return results

        if random.uniform(0, 1) > self.prob:
            return results
        indices = self.get_indexes(self.results_cache)
        mix_results = [copy.deepcopy(self.results_cache[i]) for i in indices]

        # TODO: refactor mosaic to reuse these code.
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        mosaic_masks = []
        mosaic_ignore_mask_flags = []
        with_mask = True if 'gt_masks' in results else False

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[1] * 2), int(self.img_scale[0] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        
        n_manga = 0
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(mix_results[i - 1])

            is_manga = results_patch['img_id'] > 900000000
            if is_manga:
                n_manga += 1
                if n_manga > 3:
                    continue
                im_h, im_w = results_patch['img'].shape[:2]
                if im_w > im_h and random.random() < 0.75:
                    results_patch = hcrop(results_patch, (im_h, im_w // 2), True)

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[1] / h_i,
                                self.img_scale[0] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']
            gt_ignore_mask_i = results_patch['gt_ignore_mask_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)
            mosaic_ignore_mask_flags.append(gt_ignore_mask_i)
            if with_mask and results_patch.get('gt_masks', None) is not None:

                gt_masks_i = results_patch['gt_masks']
                gt_masks_i = gt_masks_i.rescale(float(scale_ratio_i))

                gt_masks_i = translate_bitmapmask(gt_masks_i, 
                    out_shape=(int(self.img_scale[0] * 2),
                    int(self.img_scale[1] * 2)), 
                    offset_x=padw, offset_y=padh)

                # gt_masks_i = gt_masks_i.translate(
                #     out_shape=(int(self.img_scale[0] * 2),
                #                int(self.img_scale[1] * 2)),
                #     offset=padw,
                #     direction='horizontal')
                # gt_masks_i = gt_masks_i.translate(
                #     out_shape=(int(self.img_scale[0] * 2),
                #                int(self.img_scale[1] * 2)),
                #     offset=padh,
                #     direction='vertical')
                mosaic_masks.append(gt_masks_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)
        mosaic_ignore_mask_flags = np.concatenate(mosaic_ignore_mask_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * self.img_scale[1], 2 * self.img_scale[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside(
            [2 * self.img_scale[1], 2 * self.img_scale[0]]).numpy()
        
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]
        mosaic_ignore_mask_flags = mosaic_ignore_mask_flags[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags
        results['gt_ignore_mask_flags'] = mosaic_ignore_mask_flags
        

        if with_mask:
            total_instances = len(inside_inds)
            assert total_instances == np.array([m.masks.shape[0] for m in mosaic_masks]).sum()
            if total_instances > 10:
                masks = np.empty((inside_inds.sum(), mosaic_masks[0].height, mosaic_masks[0].width), dtype=np.uint8)
                msk_idx = 0
                mmsk_idx = 0
                for m in mosaic_masks:
                    for ii in range(m.masks.shape[0]):
                        if inside_inds[msk_idx]:
                            masks[mmsk_idx] = m.masks[ii]
                            mmsk_idx += 1
                        msk_idx += 1
                results['gt_masks'] = BitmapMasks(masks, mosaic_masks[0].height, mosaic_masks[0].width)
            else:
                mosaic_masks = mosaic_masks[0].cat(mosaic_masks)
                results['gt_masks'] = mosaic_masks[inside_inds]
            # assert np.all(results['gt_masks'].masks == masks) and results['gt_masks'].masks.shape == masks.shape 
        
        # assert inside_inds.sum() == results['gt_masks'].masks.shape[0]
        return results

@TRANSFORMS.register_module()
class FilterAnnotationsNoSeg(FilterAnnotations):

    def __init__(self,
                 min_gt_bbox_wh: Tuple[int, int] = (1, 1),
                 min_gt_mask_area: int = 1,
                 by_box: bool = True,
                 by_mask: bool = False,
                 keep_empty: bool = True) -> None:
        # TODO: add more filter options
        assert by_box or by_mask
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_mask_area = min_gt_mask_area
        self.by_box = by_box
        self.by_mask = by_mask
        self.keep_empty = keep_empty

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        if gt_bboxes.shape[0] == 0:
            return results

        tests = []
        if self.by_box:
            tests.append(
                ((gt_bboxes.widths > self.min_gt_bbox_wh[0]) &
                 (gt_bboxes.heights > self.min_gt_bbox_wh[1])).numpy())
                 
        if self.by_mask:
            assert 'gt_masks' in results
            gt_masks = results['gt_masks']
            tests.append(gt_masks.areas >= self.min_gt_mask_area)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        # if not keep.any():
        #     if self.keep_empty:
        #         return None

        assert len(results['gt_ignore_flags']) == len(results['gt_ignore_mask_flags'])
        keys = ('gt_bboxes', 'gt_bboxes_labels', 'gt_masks', 'gt_ignore_flags', 'gt_ignore_mask_flags')
        for key in keys:
            if key in results:
                try:
                    results[key] = results[key][keep]
                except Exception as e:
                    raise e

        return results


def hcrop(results: dict, crop_size: Tuple[int, int],
                allow_negative_crop: bool) -> Union[dict, None]:

    assert crop_size[0] > 0 and crop_size[1] > 0
    img = results['img']
    offset_h, offset_w = 0, random.choice([0, crop_size[1]])
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    # Record the homography matrix for the RandomCrop
    homography_matrix = np.array(
        [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
        dtype=np.float32)
    if results.get('homography_matrix', None) is None:
        results['homography_matrix'] = homography_matrix
    else:
        results['homography_matrix'] = homography_matrix @ results[
            'homography_matrix']

    # crop the image
    img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
    img_shape = img.shape
    results['img'] = img
    results['img_shape'] = img_shape

    # crop bboxes accordingly and clip to the image boundary
    if results.get('gt_bboxes', None) is not None:
        bboxes = results['gt_bboxes']
        bboxes.translate_([-offset_w, -offset_h])
        bboxes.clip_(img_shape[:2])
        valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
        # If the crop does not contain any gt-bbox area and
        # allow_negative_crop is False, skip this image.
        if (not valid_inds.any() and not allow_negative_crop):
            return None

        results['gt_bboxes'] = bboxes[valid_inds]

        if results.get('gt_ignore_flags', None) is not None:
            results['gt_ignore_flags'] = \
                results['gt_ignore_flags'][valid_inds]

        if results.get('gt_ignore_mask_flags', None) is not None:
            results['gt_ignore_mask_flags'] = \
                results['gt_ignore_mask_flags'][valid_inds]

        if results.get('gt_bboxes_labels', None) is not None:
            results['gt_bboxes_labels'] = \
                results['gt_bboxes_labels'][valid_inds]

        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'][
                valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
            results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                type(results['gt_bboxes']))

    # crop semantic seg
    if results.get('gt_seg_map', None) is not None:
        results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                        crop_x1:crop_x2]

    return results


@TRANSFORMS.register_module()
class RandomCropNoSeg(RandomCrop):

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:

        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_ignore_mask_flags', None) is not None:
                results['gt_ignore_mask_flags'] = \
                    results['gt_ignore_mask_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results



@TRANSFORMS.register_module()
class CachedMixUpNoSeg(CachedMixUp):

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # cache and pop images
        self.results_cache.append(copy.deepcopy(results))
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 1:
            return results

        if random.uniform(0, 1) > self.prob:
            return results

        index = self.get_indexes(self.results_cache)
        retrieve_results = copy.deepcopy(self.results_cache[index])

        # TODO: refactor mixup to reuse these code.
        if retrieve_results['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        retrieve_img = retrieve_results['img']
        with_mask = True if 'gt_masks' in results else False

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[1], self.dynamic_scale[0], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale[::-1],
                dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[1] / retrieve_img.shape[0],
                          self.dynamic_scale[0] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.ones((max(origin_h, target_h), max(
            origin_w, target_w), 3)) * self.pad_val
        padded_img = padded_img.astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes.rescale_([scale_ratio, scale_ratio])
        if with_mask:
            retrieve_gt_masks = retrieve_results['gt_masks'].rescale(
                scale_ratio)

        if self.bbox_clip_border:
            retrieve_gt_bboxes.clip_([origin_h, origin_w])

        if is_filp:
            retrieve_gt_bboxes.flip_([origin_h, origin_w],
                                     direction='horizontal')
            if with_mask:
                retrieve_gt_masks = retrieve_gt_masks.flip()

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes.translate_([-x_offset, -y_offset])
        if with_mask:

            retrieve_gt_masks = translate_bitmapmask(retrieve_gt_masks, 
                out_shape=(target_h, target_w), 
                offset_x=-x_offset, offset_y=-y_offset)

            # retrieve_gt_masks = retrieve_gt_masks.translate(
            #     out_shape=(target_h, target_w),
            #     offset=-x_offset,
            #     direction='horizontal')
            # retrieve_gt_masks = retrieve_gt_masks.translate(
            #     out_shape=(target_h, target_w),
            #     offset=-y_offset,
            #     direction='vertical')

        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes.clip_([target_h, target_w])

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']
        retrieve_gt_ignore_mask_flags = retrieve_results['gt_ignore_mask_flags']

        mixup_gt_bboxes = cp_retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], cp_retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)
        mixup_gt_ignore_mask_flags = np.concatenate(
            (results['gt_ignore_mask_flags'], retrieve_gt_ignore_mask_flags), axis=0)

        if with_mask:
            mixup_gt_masks = retrieve_gt_masks.cat(
                [results['gt_masks'], retrieve_gt_masks])

        # remove outside bbox
        inside_inds = mixup_gt_bboxes.is_inside([target_h, target_w]).numpy()
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]
        mixup_gt_ignore_flags = mixup_gt_ignore_flags[inside_inds]
        mixup_gt_ignore_mask_flags = mixup_gt_ignore_mask_flags[inside_inds]
        if with_mask:
            mixup_gt_masks = mixup_gt_masks[inside_inds]

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags
        results['gt_ignore_mask_flags'] = mixup_gt_ignore_mask_flags
        if with_mask:
            results['gt_masks'] = mixup_gt_masks
        return results