import mmcv
import argparse
import os, sys
import os.path as osp
import cv2

from anime_3dkenburns import KenBurnsPipeline, npyframes2video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anime Character Instance Segmentation')
    
    parser.add_argument('--cfg', type=str, default=None, help='KenBurns config file path')
    parser.add_argument('--input-img', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--savep', default='local_kenburns.mp4', help='video save path')
    args = parser.parse_args()

    kpipe = KenBurnsPipeline(args.cfg)

    img = mmcv.imread(args.input_img)

    kcfg = kpipe.generate_kenburns_config(img, verbose=args.verbose, savep=args.savep)

    if args.verbose:
        stage_instance_vis = kcfg.instances.draw_instances(img, draw_tags=False)
        cv2.imwrite('tmp_stage_instance.png', stage_instance_vis)

        cv2.imwrite('tmp_stage_depth_coarse.png', kcfg.stage_depth_coarse)
        cv2.imwrite('tmp_stage_depth_adjusted.png', kcfg.stage_depth_adjusted)
        cv2.imwrite('tmp_stage_depth_final.png', kcfg.stage_depth_final)

    npy_frame_list = kpipe.autozoom(kcfg, verbose=args.verbose)

    if args.verbose:
        for ii, inpainted_img in enumerate(kcfg.stage_inpainted_imgs):
            cv2.imwrite(f'tmp_stage_inpaint_{ii}.png', inpainted_img)
        for ii, mask in enumerate(kcfg.stage_inpainted_masks):
            cv2.imwrite(f'tmp_stage_inpaint_mask_{ii}.png', mask)

    npyframes2video(npy_frame_list, args.savep, playback=kcfg.playback)
