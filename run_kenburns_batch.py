import mmcv
import argparse
import os, sys
import os.path as osp
import cv2
import torch

from anime_3dkenburns import KenBurnsPipeline, npyframes2video
from utils.io_utils import find_all_imgs
from tqdm import tqdm


if __name__ == '__main__':
    # python run_kenburns_batch.py --cfg configs/3dkenburns_sam.yaml --save_dir workspace/3dkenburns_sam --input-img workspace/input_imgs
    # python run_kenburns_batch.py --cfg configs/3dkenburns.yaml --save_dir workspace/ours --input-img workspace/input_imgs
    # python run_kenburns_batch.py --cfg configs/3dkenburns.yaml --save_dir workspace/ours_marigold --input-img workspace/input_imgs
    # python run_kenburns_batch.py --cfg configs/3dkenburns_sam.yaml --save_dir workspace/sam_marigold --input-img workspace/input_imgs
    # python run_kenburns_batch.py --cfg configs/3dkenburns_sam.yaml --save_dir workspace/sam_ldm --input-img workspace/input_imgs
    # python run_kenburns_batch.py --cfg configs/3dkenburns.yaml --save_dir workspace/ours_ldm --input-img workspace/input_imgs
    parser = argparse.ArgumentParser(description='Anime Character Instance Segmentation')
    
    parser.add_argument('--cfg', type=str, default=None, help='KenBurns config file path')
    parser.add_argument('--input-img', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_dir', default='workspace', help='video save path')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    kpipe = KenBurnsPipeline(args.cfg)

    imglist = find_all_imgs(args.input_img, abs_path=True)

    with torch.inference_mode():

        for ii, imgp in enumerate(tqdm(imglist)):
            # if ii < 33:
            #     continue
            if ii == 34:
                continue
            img = mmcv.imread(imgp)

            kcfg = kpipe.generate_kenburns_config(img, verbose=args.verbose, )

            # if args.verbose:
            #     stage_instance_vis = kcfg.instances.draw_instances(img, draw_tags=False)
            #     cv2.imwrite('tmp_stage_instance.png', stage_instance_vis)

            #     cv2.imwrite('tmp_stage_depth_coarse.png', kcfg.stage_depth_coarse)
            #     cv2.imwrite('tmp_stage_depth_adjusted.png', kcfg.stage_depth_adjusted)
            #     cv2.imwrite('tmp_stage_depth_final.png', kcfg.stage_depth_final)

            npy_frame_list = kpipe.autozoom(kcfg, verbose=args.verbose)

            # if args.verbose:
                # for ii, inpainted_img in enumerate(kcfg.stage_inpainted_imgs):
                #     cv2.imwrite(f'tmp_stage_inpaint_{ii}.png', inpainted_img)
                # for ii, mask in enumerate(kcfg.stage_inpainted_masks):
                #     cv2.imwrite(f'tmp_stage_inpaint_mask_{ii}.png', mask)


            npyframes2video(npy_frame_list, osp.join(args.save_dir, osp.splitext(osp.basename(imgp))[0] + '.mp4'), playback=kcfg.playback)
