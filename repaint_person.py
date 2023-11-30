import os
import json
import base64
import io
import requests
from PIL import Image
from animeinsseg import AnimeInsSeg
import numpy as np
import argparse
import traceback
import sys
from animeinsseg.inpainting import patch_match
import cv2
import os.path as osp
from omegaconf import OmegaConf
from utils.io_utils import find_all_imgs, submit_request, img2b64, save_encoded_image


INPAINTING_FILL_METHODS = ['fill', 'original', 'latent_noise', 'latent_nothing']

def repaint_img(args, img_path, detector: AnimeInsSeg):

    prompt_default = 'masterpiece, best quality, '
    if args.prompt:
        prompt_default = prompt_default + ',' + args.prompt

    nprompt = "lowres, ((bad anatomy)), ((bad hands)), missing finger, extra digits, fewer digits, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))"
    if args.negative_prompt:
        nprompt = nprompt + ", " + args.negative_prompt

    assert args.fill in INPAINTING_FILL_METHODS, \
        f'Fill method must be one of {INPAINTING_FILL_METHODS}.'

    resolution = args.resolution
    img = Image.open(img_path).convert('RGB')
    imgname = osp.basename(img_path)
    if img.height > img.width:
        W = resolution
        H = (img.height / img.width * resolution) // 32 * 32
        H = int(H)
    else:
        H = resolution
        W = (img.width / img.height * resolution) // 32 * 32 
        W = int(W)

    if args.to_grey:
        img = img.convert('L').convert('RGB')

    options_shared = {
        'width': W,
        'height': H,
        'steps': args.steps,
        'cfg_scale': args.cfg_scale,
        'sample_name': args.sample_name,
        'denoising_strength': args.denoising_strength,
        "alwayson_scripts": {
            "controlnet": {
            "args": [
                {
                    "input_image": "",
                    "module": "lineart_anime",
                    "model": "control_v11p_sd15s2_lineart_anime [3825e83e]",
                    "weight": 1,
                    "resize_mode": "Inner Fit (Scale to Fit)",
                    "lowvram": False,
                    "processor_res": resolution,
                    "threshold_a": 64,
                    "threshold_b": 64,
                    "guidance": 1,
                    "guidance_start": 0,
                    "guidance_end": 1,
                    "guessmode": False,
                    "pixel_perfect": True
                }
            ]
            }
        },
    }

    img = img.resize((W, H), resample=Image.Resampling.LANCZOS)

    if not args.onebyone:
        # detected tags got mixed during the rendering processing
        # imgbgr = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        # tags, character_tags = detector.tagger.label_cv2_bgr(imgbgr)
        prompt = prompt_default
        # prompt = prompt_default
        
        img_b64 = img2b64(img)
        options_shared['alwayson_scripts']['controlnet']['args'][0]['input_image'] = img_b64
        data = {
            "init_images": [img_b64],
            "prompt": prompt,
            "negative_prompt": nprompt,
            **options_shared
        }
        data = json.dumps(data)

        print('runing default cldm ...')
        print(f'prompt: {prompt} \n negative prompt: {nprompt}')
        response = submit_request(args.url, data)
        output_img_b64 = response.json()['images'][0]

        save_encoded_image(output_img_b64, osp.join(save_dir, 'repaint-default-' + imgname+'.png'))
        return

    instances = detector.infer(img_path, output_type='numpy', infer_grey=args.to_grey)
    instances.remove_duplicated()
    im_cv = cv2.imread(img_path)
    ins_drawed = instances.draw_instances(im_cv, draw_bbox=False, mask_alpha=0.35, draw_tags=False)
    cv2.imwrite(osp.join(save_dir, imgname+'_instances.png'), ins_drawed)

    mask_fg = None
    masks = []
    if not instances.is_empty:
        for mask in instances.masks:
            mask = cv2.resize(mask.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_AREA)
            mask = Image.fromarray(mask)
            masks.append(mask)
        mask_fg = cv2.resize(instances.compose_masks().astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_AREA)

    if mask_fg is not None:
        bg = patch_match.inpaint(np.array(img), mask_fg, patch_size=3)
        bg_tags, character_tags = detector.tagger.label_cv2_bgr(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        bg_prompt = ','.join(bg_tags) + ',' + prompt_default
        bg_nprompt = nprompt + ',' + args.bg_nprompt
        bg_prompt = bg_prompt.replace('portrait', '')
        bg = Image.fromarray(bg)
        bg_b64 = img2b64(bg)

        options_shared['alwayson_scripts']['controlnet']['args'][0]['input_image'] = bg_b64
        data = {
            "init_images": [bg_b64],
            "prompt": bg_prompt,
            "negative_prompt": bg_nprompt,
            **options_shared
        }
        data = json.dumps(data)

        print('fg...')
        print(f'bg prompt: {bg_prompt} \nbg negative prompt: {bg_nprompt}')
        response = submit_request(args.url, data)
        output_img_b64 = response.json()['images'][0]

        if args.save_intermediate:
            save_encoded_image(output_img_b64, osp.join(save_dir, 'repaint-bg-' + imgname+'.png'))
        bg_repainted = Image.open(io.BytesIO(base64.b64decode(output_img_b64)))
        img_bg_repainted = Image.composite(img, bg_repainted, Image.fromarray(mask_fg))
        # img_bg_repainted.save(osp.join(save_dir, 'repaint-bg-composed-' + imgname+'.png'))
        img = img_bg_repainted
    
    if instances.is_empty:
        return

    img_b64 = None
    num_fg = len(masks)
    for ii, mask in enumerate(masks):
        if img_b64 is None:
            img_b64 = img2b64(img)
        mask_b64 = img2b64(mask)
        tags = instances.tags[ii]
        prompt = tags.replace(' ', ',') + ',' + prompt_default
        # prompt = prompt_default

        options_shared['alwayson_scripts']['controlnet']['args'][0]['input_image'] = img_b64
        request = {
            "init_images": [img_b64],
            "mask": mask_b64,
            "prompt": prompt,
            "negative_prompt": nprompt,
            'mask_blur': args.mask_blur,
            'inpainting_fill': INPAINTING_FILL_METHODS.index(args.fill),
            'inpaint_full_res': args.inpaint_full_res,
            'inpaint_full_res_padding': args.inpaint_full_res_padding,
            **options_shared
        }

        data = json.dumps(request)
        print(f'runing fg repainting...{ii+1}/{num_fg}')
        print(f'fg prompt: {prompt} \n fg negative prompt: {nprompt}')
        response = submit_request(args.url, data)
        img_b64 = response.json()['images'][0]
        if args.save_intermediate or ii == num_fg - 1:
            save_encoded_image(img_b64, osp.join(save_dir, f'repaint-fg{ii}-' + imgname+'.png'))
        mask.save(osp.join(save_dir, f'repaint-fg{ii}-mask-' + imgname+'.png'))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Inpaint instances of people using stable '
                                                 'diffusion.')
    parser.add_argument('--img_path', type=str, default='', help='Path to input image.')
    parser.add_argument('--onebyone', type=bool, default=True, help='repainting person one by one')
    parser.add_argument('-p', '--prompt', type=str, default='',
                        help='Stable diffusion prompt to use.')
    parser.add_argument('-n', '--negative_prompt', type=str, default='',
                        help='Stable diffusion negative prompt.')
    parser.add_argument('-W', '--width', type=int, default=768, help='Width of output image.')
    parser.add_argument('-H', '--height', type=int, default=768, help='Height of output image.')
    parser.add_argument('-s', '--steps', type=int, default=24, help='Number of diffusion steps.')
    parser.add_argument('-c', '--cfg_scale', type=int, default=9, help='Classifier free guidance '
                        'scale, i.e. how strongly the image should conform to prompt.')
    parser.add_argument('-S', '--sample_name', type=str, default='Euler a', help='Name of sampler '
                        'to use.')
    parser.add_argument('-d', '--denoising_strength', type=float, default=0.75, help='How much to '
                        'disregard original image.')
    parser.add_argument('-f', '--fill', type=str, default=INPAINTING_FILL_METHODS[1],
                        help='The fill method to use for inpainting.')
    parser.add_argument('-b', '--mask_blur', type=int, default=4, help='Blur radius of Gaussian '
                        'filter to apply to mask.')
    parser.add_argument('-r', '--resolution', type=int, default=640, help='inpainting resolution')
    parser.add_argument('--save_dir', type=str, default='repaint_output')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:7860/sdapi/v1/img2img', help='img2img url')
    parser.add_argument('--cfg', type=str, default='', help='repaint config path')
    parser.add_argument('--bg_nprompt', type=str, default='((person)), character, 1girl, 1boy', help='background negative prompt')
    parser.add_argument('--inpaint_full_res', type=int, default=1)
    parser.add_argument('--inpaint_full_res_padding', type=int, default=32)
    parser.add_argument('--detector_ckpt', type=str, default='models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt')
    parser.add_argument('--save_intermediate', type=bool, default=False)
    parser.add_argument('--to-grey', type=bool, default=False)
    parser.add_argument('--infer-tagger', type=bool, default=True)
    args = parser.parse_args()

    if osp.exists(args.cfg):
        args = OmegaConf.create(vars(args))
        args.merge_with(OmegaConf.load(args.cfg))
        print(args)

    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    detector = AnimeInsSeg(args.detector_ckpt, device='cuda')
    detector.init_tagger()
    imgp_list = args.img_path
    if isinstance(imgp_list, str):
        if osp.isdir(imgp_list):
            imgp_list = find_all_imgs(imgp_list, abs_path=True)
        else:
            imgp_list = [imgp_list]
    for imgp in imgp_list:
        repaint_img(args, imgp, detector)
    



