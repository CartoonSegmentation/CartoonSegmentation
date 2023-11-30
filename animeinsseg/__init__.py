import mmcv, torch
from tqdm import tqdm
from einops import rearrange
import os
import os.path as osp
import cv2
import gc
import math

from .anime_instances import AnimeInstances
import numpy as np
from typing import List, Tuple, Union, Optional, Callable
from mmengine import Config
from mmengine.model.utils import revert_sync_batchnorm
from mmdet.utils import register_all_modules, get_test_pipeline_cfg
from mmdet.apis import init_detector
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox.transforms import scale_boxes, get_box_wh
from mmdet.models.dense_heads.rtmdet_ins_head import RTMDetInsHead
from pycocotools.coco import COCO
from mmcv.transforms import Compose
from mmdet.models.detectors.single_stage import SingleStageDetector

from utils.logger import LOGGER
from utils.io_utils import square_pad_resize, find_all_imgs, imglist2grid, mask2rle, dict2json, scaledown_maxsize, resize_pad
from utils.constants import DEFAULT_DEVICE, CATEGORIES
from utils.booru_tagger import Tagger

from .models.animeseg_refine import AnimeSegmentation, load_refinenet, get_mask
from .models.rtmdet_inshead_custom import RTMDetInsSepBNHeadCustom

from torchvision.ops.boxes import box_iou
import torch.nn.functional as F


def prepare_refine_batch(segmentations: np.ndarray, img: np.ndarray, max_batch_size: int = 4, device: str = 'cpu', input_size: int = 720):

    img, (pt, pb, pl, pr) = resize_pad(img, input_size, pad_value=(0, 0, 0))

    img = img.transpose((2, 0, 1)).astype(np.float32) / 255.

    batch = []
    num_seg = len(segmentations)
    
    for ii, seg in enumerate(segmentations):
        seg, _ = resize_pad(seg, input_size, 0)
        seg = seg[None, ...]
        batch.append(np.concatenate((img, seg)))

        if ii == num_seg - 1:
            yield torch.from_numpy(np.array(batch)).to(device), (pt, pb, pl, pr)
        elif len(batch) >= max_batch_size:
            yield torch.from_numpy(np.array(batch)).to(device), (pt, pb, pl, pr)
            batch = []


VALID_REFINEMETHODS = {'animeseg', 'none'}

register_all_modules()


def single_image_preprocess(img: Union[str, np.ndarray], pipeline: Compose):
    if isinstance(img, str):
        img = mmcv.imread(img)
    elif not isinstance(img, np.ndarray):
        raise NotImplementedError

    # img = square_pad_resize(img, 1024)[0]

    data_ = dict(img=img, img_id=0)
    data_ = pipeline(data_)
    data_['inputs'] = [data_['inputs']]
    data_['data_samples'] = [data_['data_samples']]

    return data_, img

def animeseg_refine(det_pred: DetDataSample, img: np.ndarray, net: AnimeSegmentation, to_rgb=True, input_size: int = 1024):
    
    num_pred = len(det_pred.pred_instances)
    if num_pred < 1:
        return
    
    with torch.no_grad():
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg_thr = 0.5
        mask = get_mask(net, img, s=input_size)[..., 0]
        mask = (mask > seg_thr)
        
        ins_masks = det_pred.pred_instances.masks

        if isinstance(ins_masks, torch.Tensor):
            tensor_device = ins_masks.device
            tensor_dtype = ins_masks.dtype
            to_tensor = True
            ins_masks = ins_masks.cpu().numpy()

        area_original = np.sum(ins_masks, axis=(1, 2))
        masks_refined = np.bitwise_and(ins_masks, mask[None, ...])
        area_refined = np.sum(masks_refined, axis=(1, 2))

        for ii in range(num_pred):
            if area_refined[ii] / area_original[ii] > 0.3:
                ins_masks[ii] = masks_refined[ii]
        ins_masks = np.ascontiguousarray(ins_masks)

        # for ii, insm in enumerate(ins_masks):
        #     cv2.imwrite(f'{ii}.png', insm.astype(np.uint8) * 255)

        if to_tensor:
            ins_masks = torch.from_numpy(ins_masks).to(dtype=tensor_dtype).to(device=tensor_device)

        det_pred.pred_instances.masks = ins_masks
        # rst = np.concatenate((mask * img + 1 - mask, mask * 255), axis=2).astype(np.uint8)
        # cv2.imwrite('rst.png', rst)


# def refinenet_forward(det_pred: DetDataSample, img: np.ndarray, net: AnimeSegmentation, to_rgb=True, input_size: int = 1024):
    
#     num_pred = len(det_pred.pred_instances)
#     if num_pred < 1:
#         return
    
#     with torch.no_grad():
#         if to_rgb:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         seg_thr = 0.5

#         h0, w0 = h, w = img.shape[0], img.shape[1]
#         if h > w:
#             h, w = input_size, int(input_size * w / h)
#         else:
#             h, w = int(input_size * h / w), input_size
#         ph, pw = input_size - h, input_size - w
#         tmpImg = np.zeros([s, s, 3], dtype=np.float32)
#         tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h)) / 255
#         tmpImg = tmpImg.transpose((2, 0, 1))
#         tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(model.device)
#         with torch.no_grad():
#             if use_amp:
#                 with amp.autocast():
#                     pred = model(tmpImg)
#                 pred = pred.to(dtype=torch.float32)
#             else:
#                 pred = model(tmpImg)
#             pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
#             pred = cv2.resize(pred.cpu().numpy().transpose((1, 2, 0)), (w0, h0))[:, :, np.newaxis]
#             return pred

#         mask = (mask > seg_thr)
        
#         ins_masks = det_pred.pred_instances.masks

#         if isinstance(ins_masks, torch.Tensor):
#             tensor_device = ins_masks.device
#             tensor_dtype = ins_masks.dtype
#             to_tensor = True
#             ins_masks = ins_masks.cpu().numpy()

#         area_original = np.sum(ins_masks, axis=(1, 2))
#         masks_refined = np.bitwise_and(ins_masks, mask[None, ...])
#         area_refined = np.sum(masks_refined, axis=(1, 2))

#         for ii in range(num_pred):
#             if area_refined[ii] / area_original[ii] > 0.3:
#                 ins_masks[ii] = masks_refined[ii]
#         ins_masks = np.ascontiguousarray(ins_masks)

#         # for ii, insm in enumerate(ins_masks):
#         #     cv2.imwrite(f'{ii}.png', insm.astype(np.uint8) * 255)

#         if to_tensor:
#             ins_masks = torch.from_numpy(ins_masks).to(dtype=tensor_dtype).to(device=tensor_device)

#         det_pred.pred_instances.masks = ins_masks


def read_imglst_from_txt(filep) -> List[str]:
    with open(filep, 'r', encoding='utf8') as f:
        lines = f.read().splitlines() 
    return lines


class AnimeInsSeg:

    def __init__(self, ckpt: str, default_det_size: int = 640, device: str = None, 
                 refine_kwargs: dict = {'refine_method': 'refinenet_isnet'},
                 tagger_path: str = 'models/wd-v1-4-swinv2-tagger-v2/model.onnx', mask_thr=0.3) -> None:
        self.ckpt = ckpt
        self.default_det_size = default_det_size
        self.device = DEFAULT_DEVICE if device is None else device

        # init detector in mmdet's way

        ckpt = torch.load(ckpt, map_location='cpu')
        cfg = Config.fromstring(ckpt['meta']['cfg'].replace('file_client_args', 'backend_args'), file_format='.py')
        cfg.visualizer = []
        cfg.vis_backends = {}
        cfg.default_hooks.pop('visualization')
        

        # self.model: SingleStageDetector = init_detector(cfg, checkpoint=None, device='cpu')
        model = MODELS.build(cfg.model)
        model = revert_sync_batchnorm(model)

        self.model = model.to(self.device).eval()
        self.model.load_state_dict(ckpt['state_dict'], strict=False)
        self.model = self.model.to(self.device).eval()
        self.cfg = cfg.copy()

        test_pipeline = get_test_pipeline_cfg(self.cfg.copy())
        test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(test_pipeline)
        self.default_data_pipeline = test_pipeline

        self.refinenet = None
        self.refinenet_animeseg: AnimeSegmentation = None
        self.postprocess_refine: Callable = None

        if refine_kwargs is not None:
            self.set_refine_method(**refine_kwargs)

        self.tagger = None
        self.tagger_path = tagger_path

        self.mask_thr = mask_thr

    def init_tagger(self, tagger_path: str = None):
        tagger_path = self.tagger_path if tagger_path is None else tagger_path
        self.tagger = Tagger(self.tagger_path)

    def infer_tags(self, instances: AnimeInstances, img: np.ndarray, infer_grey: bool = False):
        if self.tagger is None:
            self.init_tagger()

        if infer_grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None][..., [0, 0, 0]]

        num_ins = len(instances)
        for ii in range(num_ins):
            bbox = instances.bboxes[ii]
            mask = instances.masks[ii]
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.cpu().numpy()
                mask = mask.cpu().numpy()
            bbox = bbox.astype(np.int32)
            
            crop = img[bbox[1]: bbox[3] + bbox[1], bbox[0]: bbox[2] + bbox[0]].copy()
            mask = mask[bbox[1]: bbox[3] + bbox[1], bbox[0]: bbox[2] + bbox[0]]
            crop[mask == 0] = 255
            tags, character_tags = self.tagger.label_cv2_bgr(crop)
            exclude_tags = ['simple_background', 'white_background']
            valid_tags = []
            for tag in tags:
                if tag in exclude_tags:
                    continue
                valid_tags.append(tag)
            instances.tags[ii] = ' '.join(valid_tags)
            instances.character_tags[ii] = character_tags

    @torch.no_grad()
    def infer_embeddings(self, imgs, det_size = None):

        def hijack_bbox_mask_post_process(
                self,
                results,
                mask_feat,
                cfg,
                rescale: bool = False,
                with_nms: bool = True,
                img_meta: Optional[dict] = None):

            stride = self.prior_generator.strides[0][0]
            if rescale:
                assert img_meta.get('scale_factor') is not None
                scale_factor = [1 / s for s in img_meta['scale_factor']]
                results.bboxes = scale_boxes(results.bboxes, scale_factor)

            if hasattr(results, 'score_factors'):
                # TODO： Add sqrt operation in order to be consistent with
                #  the paper.
                score_factors = results.pop('score_factors')
                results.scores = results.scores * score_factors

            # filter small size bboxes
            if cfg.get('min_bbox_size', -1) >= 0:
                w, h = get_box_wh(results.bboxes)
                valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
                if not valid_mask.all():
                    results = results[valid_mask]

            # results.mask_feat = mask_feat
            return results, mask_feat

        def hijack_detector_predict(self: SingleStageDetector,
                    batch_inputs: torch.Tensor,
                    batch_data_samples: SampleList,
                    rescale: bool = True) -> SampleList:
            x = self.extract_feat(batch_inputs)

            bbox_head: RTMDetInsSepBNHeadCustom = self.bbox_head
            old_postprocess = RTMDetInsSepBNHeadCustom._bbox_mask_post_process
            RTMDetInsSepBNHeadCustom._bbox_mask_post_process = hijack_bbox_mask_post_process
            # results_list = bbox_head.predict(
            #     x, batch_data_samples, rescale=rescale)
            
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]

            outs = bbox_head(x)

            results_list = bbox_head.predict_by_feat(
                *outs, batch_img_metas=batch_img_metas, rescale=rescale)

            # batch_data_samples = self.add_pred_to_datasample(
            #     batch_data_samples, results_list)
            
            RTMDetInsSepBNHeadCustom._bbox_mask_post_process = old_postprocess
            return results_list

        old_predict = SingleStageDetector.predict
        SingleStageDetector.predict = hijack_detector_predict
        test_pipeline, imgs, _ = self.prepare_data_pipeline(imgs, det_size)

        if len(imgs) > 1:
            imgs = tqdm(imgs)
        model = self.model
        img = imgs[0]
        data_, img = test_pipeline(img)
        data = model.data_preprocessor(data_, False)
        instance_data, mask_feat = model(**data, mode='predict')[0]
        SingleStageDetector.predict = old_predict

        # print((instance_data.scores > 0.9).sum())
        return img, instance_data, mask_feat

    def segment_with_bboxes(self, img, bboxes: torch.Tensor, instance_data, mask_feat: torch.Tensor):
        # instance_data.bboxes: x1, y1, x2, y2
        maxidx = torch.argmax(instance_data.scores)
        bbox = instance_data.bboxes[maxidx].cpu().numpy()
        p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        tgt_bboxes = instance_data.bboxes

        im_h, im_w = img.shape[:2]
        long_side = max(im_h, im_w)
        bbox_head: RTMDetInsSepBNHeadCustom = self.model.bbox_head
        priors, kernels = instance_data.priors, instance_data.kernels
        stride = bbox_head.prior_generator.strides[0][0]

        ins_bboxes, ins_segs, scores = [], [], []
        for bbox in bboxes:
            bbox = torch.from_numpy(np.array([bbox])).to(tgt_bboxes.dtype).to(tgt_bboxes.device)
            ioulst = box_iou(bbox, tgt_bboxes).squeeze()
            matched_idx = torch.argmax(ioulst)

            mask_logits = bbox_head._mask_predict_by_feat_single(
                mask_feat, kernels[matched_idx][None, ...], priors[matched_idx][None, ...])

            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0), scale_factor=stride, mode='bilinear')

            mask_logits = F.interpolate(
                mask_logits,
                size=[long_side, long_side],
                mode='bilinear',
                align_corners=False)[..., :im_h, :im_w]
            mask = mask_logits.sigmoid().squeeze()
            mask = mask > 0.5
            mask = mask.cpu().numpy()
            ins_segs.append(mask)
            
            matched_iou_score = ioulst[matched_idx]
            matched_score = instance_data.scores[matched_idx]
            scores.append(matched_score.cpu().item())
            matched_bbox = tgt_bboxes[matched_idx]

            ins_bboxes.append(matched_bbox.cpu().numpy())
            # p1, p2 = (int(matched_bbox[0]), int(matched_bbox[1])), (int(matched_bbox[2]), int(matched_bbox[3]))

        if len(ins_bboxes) > 0:
            ins_bboxes = np.array(ins_bboxes).astype(np.int32)
            ins_bboxes[:, 2:] -= ins_bboxes[:, :2]
            ins_segs = np.array(ins_segs)
        instances = AnimeInstances(ins_segs, ins_bboxes, scores)
        
        self._postprocess_refine(instances, img)
        drawed = instances.draw_instances(img)
        # cv2.imshow('drawed', drawed)
        # cv2.waitKey(0)
        
        return instances

    def set_detect_size(self, det_size: Union[int, Tuple]):
        if isinstance(det_size, int):
            det_size = (det_size, det_size)
        self.default_data_pipeline.transforms[1].scale = det_size
        self.default_data_pipeline.transforms[2].size = det_size
        
    @torch.no_grad()
    def infer(self, imgs: Union[List, str, np.ndarray], 
              pred_score_thr: float = 0.3,
              refine_kwargs: dict = None,
              output_type: str="tensor", 
              det_size: int = None, 
              save_dir: str = '',
              save_visualization: bool = False,
              save_annotation: str = '',
              infer_tags: bool = False,
              obj_id_start: int = -1, 
              img_id_start: int = -1,
              verbose: bool = False,
              infer_grey: bool = False,
              save_mask_only: bool = False,
              val_dir=None,
              max_instances: int = 100,
              **kwargs) -> Union[List[AnimeInstances], AnimeInstances, None]:
    
        """
        Args:
            imgs (str, ndarray, Sequence[str/ndarray]):
                Either image files or loaded images.

        Returns:
            :obj:`AnimeInstances` or list[:obj:`AnimeInstances`]:
            If save_annotation or save_annotation, return None.
        """

        if det_size is not None:
            self.set_detect_size(det_size)
        if refine_kwargs is not None:
            self.set_refine_method(**refine_kwargs)

        self.set_max_instance(max_instances)

        if isinstance(imgs, str):
            if imgs.endswith('.txt'):
                imgs = read_imglst_from_txt(imgs)
        
        if save_annotation or save_visualization:
            return self._infer_save_annotations(imgs, pred_score_thr, det_size, save_dir, save_visualization, \
                                               save_annotation, infer_tags, obj_id_start, img_id_start, val_dir=val_dir)
        else:
            return self._infer_simple(imgs, pred_score_thr, det_size, output_type, infer_tags, verbose=verbose, infer_grey=infer_grey)
        
    def _det_forward(self, img, test_pipeline, pred_score_thr: float = 0.3) -> Tuple[AnimeInstances, np.ndarray]:
        data_, img = test_pipeline(img)
        with torch.no_grad():
            results: DetDataSample = self.model.test_step(data_)[0]
            pred_instances = results.pred_instances
            pred_instances = pred_instances[pred_instances.scores > pred_score_thr]
            if len(pred_instances) < 1:
                return AnimeInstances(), img
        
        del data_
        
        bboxes = pred_instances.bboxes.to(torch.int32)
        bboxes[:, 2:] -= bboxes[:, :2]
        masks = pred_instances.masks
        scores = pred_instances.scores
        return AnimeInstances(masks, bboxes, scores), img
        
    def _infer_simple(self, imgs: Union[List, str, np.ndarray], 
                      pred_score_thr: float = 0.3,
                      det_size: int = None,
                      output_type: str = "tensor",
                      infer_tags: bool = False,
                      infer_grey: bool = False,
                      verbose: bool = False) -> Union[DetDataSample, List[DetDataSample]]:
        
        if isinstance(imgs, List):
            return_list = True
        else:
            return_list = False

        assert output_type in {'tensor', 'numpy'}

        test_pipeline, imgs, _ = self.prepare_data_pipeline(imgs, det_size)
        predictions = []

        if len(imgs) > 1:
            imgs = tqdm(imgs)

        for img in imgs:
            instances, img = self._det_forward(img, test_pipeline, pred_score_thr)
            # drawed = instances.draw_instances(img)
            # cv2.imwrite('drawed.jpg', drawed)
            self.postprocess_results(instances, img)
            # drawed = instances.draw_instances(img)
            # cv2.imwrite('drawed_post.jpg', drawed)

            if infer_tags:
                self.infer_tags(instances, img, infer_grey)
                
            if output_type == 'numpy':
                instances.to_numpy()
                
            predictions.append(instances)

        if return_list:
            return predictions
        else:
            return predictions[0]

    def _infer_save_annotations(self, imgs: Union[List, str, np.ndarray], 
              pred_score_thr: float = 0.3,
              det_size: int = None, 
              save_dir: str = '',
              save_visualization: bool = False,
              save_annotation: str = '',
              infer_tags: bool = False,
              obj_id_start: int = 100000000000, 
              img_id_start: int = 100000000000,
              save_mask_only: bool = False,
              val_dir = None,
              **kwargs) -> None:

        coco_api = None
        if isinstance(imgs, str) and imgs.endswith('.json'):
            coco_api = COCO(imgs)

            if val_dir is None:
                val_dir = osp.join(osp.dirname(osp.dirname(imgs)), 'val')
            imgs = coco_api.getImgIds()
            imgp2ids = {}
            imgps, coco_imgmetas = [], []
            for imgid in imgs:
                imeta = coco_api.loadImgs(imgid)[0]
                imgname = imeta['file_name']
                imgp = osp.join(val_dir, imgname)
                imgp2ids[imgp] = imgid
                imgps.append(imgp)
                coco_imgmetas.append(imeta)
            imgs = imgps

        test_pipeline, imgs, target_dir = self.prepare_data_pipeline(imgs, det_size)
        if save_dir == '':
            save_dir = osp.join(target_dir, \
                osp.basename(self.ckpt).replace('.ckpt', '').replace('.pth', '').replace('.pt', ''))
            
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        det_annotations = []
        image_meta = []
        obj_id = obj_id_start + 1
        image_id = img_id_start + 1

        for ii, img in enumerate(tqdm(imgs)):
            # prepare data
            if isinstance(img, str):
                img_name = osp.basename(img)
            else:
                img_name = f'{ii}'.zfill(12) + '.jpg'

            if coco_api is not None:
                image_id = imgp2ids[img]
            
            try:
                instances, img = self._det_forward(img, test_pipeline, pred_score_thr)
            except Exception as e:
                raise e
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    try:
                        instances, img = self._det_forward(img, test_pipeline, pred_score_thr)
                    except:
                        LOGGER.warning(f'cuda out of memory: {img_name}')
                        if isinstance(img, str):
                            img = cv2.imread(img)
                        instances = None

            if instances is not None:
                self.postprocess_results(instances, img)

                if infer_tags:
                    self.infer_tags(instances, img)

                if save_visualization:
                    out_file = osp.join(save_dir, img_name)
                    self.save_visualization(out_file, img, instances)

            if save_annotation:
                im_h, im_w = img.shape[:2]
                image_meta.append({
                    "id": image_id,"height": im_h,"width": im_w, 
                    "file_name": img_name, "id": image_id
                })
                if instances is not None:
                    for ii in range(len(instances)):
                        segmentation = instances.masks[ii].squeeze().cpu().numpy().astype(np.uint8)
                        area = segmentation.sum()
                        segmentation *= 255
                        if save_mask_only:
                            cv2.imwrite(osp.join(save_dir, 'mask_' + str(ii).zfill(3) + '_' +img_name+'.png'), segmentation)
                        else:
                            score = instances.scores[ii]
                            if isinstance(score, torch.Tensor):
                                score = score.item()
                            score = float(score)
                            bbox = instances.bboxes[ii].cpu().numpy()
                            bbox = bbox.astype(np.float32).tolist()
                            segmentation = mask2rle(segmentation)
                            tag_string = instances.tags[ii]
                            tag_string_character = instances.character_tags[ii]
                            det_annotations.append({'id': obj_id, 'category_id': 0, 'iscrowd': 0, 'score': score,
                                'segmentation': segmentation, 'image_id': image_id, 'area': area,
                                'tag_string': tag_string, 'tag_string_character': tag_string_character, 'bbox': bbox
                            })
                        obj_id += 1
                image_id += 1

        if save_annotation != '' and not save_mask_only:
            det_meta = {"info": {},"licenses": [], "images": image_meta, 
                        "annotations": det_annotations, "categories": CATEGORIES}
            detp = save_annotation
            dict2json(det_meta, detp)
            LOGGER.info(f'annotations saved to {detp}')
    
    def set_refine_method(self, refine_method: str = 'none', refine_size: int = 720):
        if refine_method == 'none':
            self.postprocess_refine = None
        elif refine_method == 'animeseg':
            if self.refinenet_animeseg is None:
                self.refinenet_animeseg = load_refinenet(refine_method)
            self.postprocess_refine = lambda det_pred, img: \
                                        animeseg_refine(det_pred, img, self.refinenet_animeseg, True, refine_size)
        elif refine_method == 'refinenet_isnet':
            if self.refinenet is None:
                self.refinenet = load_refinenet(refine_method)
            self.postprocess_refine = self._postprocess_refine
        else:
            raise NotImplementedError(f'Invalid refine method: {refine_method}')
        
    def _postprocess_refine(self, instances: AnimeInstances, img: np.ndarray, refine_size: int = 720, max_refine_batch: int = 4, **kwargs):
        
        if instances.is_empty:
            return
        
        segs = instances.masks
        is_tensor = instances.is_tensor
        if is_tensor:
            segs = segs.cpu().numpy()
        segs = segs.astype(np.float32)
        im_h, im_w = img.shape[:2]
        
        masks = []
        with torch.no_grad():
            for batch, (pt, pb, pl, pr) in prepare_refine_batch(segs, img, max_refine_batch, self.device, refine_size):
                preds = self.refinenet(batch)[0][0].sigmoid()
                if pb == 0:
                    pb = -im_h
                if pr == 0:
                    pr = -im_w
                preds = preds[..., pt: -pb, pl: -pr]
                preds  = torch.nn.functional.interpolate(preds, (im_h, im_w), mode='bilinear', align_corners=True)
                masks.append(preds.cpu()[:, 0])

        masks = (torch.concat(masks, dim=0) > self.mask_thr).to(self.device)
        if not is_tensor:
            masks = masks.cpu().numpy()
        instances.masks = masks


    def prepare_data_pipeline(self, imgs: Union[str, np.ndarray, List], det_size: int) -> Tuple[Compose, List, str]:
        
        if det_size is None:
            det_size = self.default_det_size

        target_dir = './workspace/output'
        # cast imgs to a list of np.ndarray or image_file_path  if necessary
        if isinstance(imgs, str):
            if osp.isdir(imgs):
                target_dir = imgs
                imgs = find_all_imgs(imgs, abs_path=True)
            elif osp.isfile(imgs):
                target_dir = osp.dirname(imgs)
                imgs = [imgs]
        elif isinstance(imgs, np.ndarray) or isinstance(imgs, str):
            imgs = [imgs]
        elif isinstance(imgs, List):
            if len(imgs) > 0:
                if isinstance(imgs[0], np.ndarray) or isinstance(imgs[0], str):
                    pass
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        test_pipeline = lambda img: single_image_preprocess(img, pipeline=self.default_data_pipeline)
        return test_pipeline, imgs, target_dir

    def save_visualization(self, out_file: str, img: np.ndarray, instances: AnimeInstances):
        drawed = instances.draw_instances(img)
        mmcv.imwrite(drawed, out_file)
    
    def postprocess_results(self, results: DetDataSample, img: np.ndarray) -> None:
        if self.postprocess_refine is not None:
            self.postprocess_refine(results, img)

    def set_mask_threshold(self, mask_thr: float):
        self.model.bbox_head.test_cfg['mask_thr_binary'] = mask_thr

    def set_max_instance(self, num_ins):
        self.model.bbox_head.test_cfg['max_per_img'] = num_ins