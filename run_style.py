import os
import json
import base64
import io
import requests
from PIL import Image
import numpy as np
import argparse
import random
from tqdm import tqdm
import os.path as osp
from omegaconf import OmegaConf
from utils.io_utils import find_all_imgs, submit_request, img2b64, save_encoded_image
from random import randint
from requests.auth import HTTPBasicAuth
import cv2
from copy import deepcopy
from pathlib import Path
import math


INPAINTING_FILL_METHODS = ['fill', 'original', 'latent_noise', 'latent_nothing']


def run_sdinpaint(img: Image.Image, mask: Image.Image, data: dict, prompt: str = '', nprompt: str = '', url='', auth=None) -> str:
    if isinstance(img, Image.Image):
        img_b64 = img2b64(img)
    else:
        assert isinstance(img, str)
        img_b64 = img
    mask_b64 = img2b64(mask)
    data['init_images'] = [img_b64]
    data['alwayson_scripts']['controlnet']['args'][0]['input_image'] = img_b64
    data['mask'] = mask_b64
    data['negative_prompt'] = nprompt
    data['prompt'] = prompt

    response = submit_request(url, json.dumps(data), auth=auth)
    img_b64 = response.json()['images'][0]
    return img_b64

def long_side_to(H, W, long_side):
    asp = H / W
    if asp > 1:
        H = long_side
        H = int(round(H / 32)) * 32
        W = int(round(H / asp / 32)) * 32
    else:
        W = long_side
        W = int(round(W / 32)) * 32
        H = int(round(W * asp / 32)) * 32
    return H, W

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Inpaint instances of people using stable '
                                                 'diffusion.')
    # workspace\forpaper\eval\kenburns\original\1.png
    parser.add_argument('--img_path', type=str, help='Path to input image. Can be directory.')
    # parser.add_argument('--img_path', type=str, default=r'workspace/style/original/', help='Path to input image.')
    parser.add_argument('--onebyone', type=bool, default=True, help='repainting person one by one')
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
    parser.add_argument('--url', type=str, default='http://127.0.0.1:7860/sdapi/v1/txt2img', help='img2img url')
    parser.add_argument('--cfg', type=str, default='configs/3d_pixar.yaml', help='repaint config path')
    parser.add_argument('--bg_nprompt', type=str, default='', help='background negative prompt')
    parser.add_argument('--inpaint_full_res', type=int, default=1)
    parser.add_argument('--inpaint_full_res_padding', type=int, default=32)
    parser.add_argument('--detector_ckpt', type=str, default='models/AnimeInstanceSegmentation/rtmdetl_e60.ckpt')
    parser.add_argument('--save_intermediate', type=bool, default=False)
    parser.add_argument('--to-grey', type=bool, default=False)
    parser.add_argument('--infer-tagger', type=bool, default=True)
    parser.add_argument('--style-prompt', default='')
    parser.add_argument('--global-nprompt', default='')
    parser.add_argument('--apply-bg-tagger', default=False)
    parser.add_argument('--apply-fg-tagger', default=True)

    args = parser.parse_args()

    args = OmegaConf.create(vars(args))
    args.merge_with(OmegaConf.load(args.cfg))

    data = {
        **OmegaConf.to_container(args.sd_params),
        # "init_images": [img_b64]
    }

    auth = None
    if 'username' in args:
        username = args.pop('username')
        password = args.pop('password')
        auth = HTTPBasicAuth(username, password)

    img_path = args.img_path
    if osp.isfile(img_path):
        imglist = [img_path]
    else:
        imglist = find_all_imgs(img_path, abs_path=True)
        imglist = imglist[::-1]
    
    detector = None
    for ii, img_path in enumerate(imglist):
        print(f'repainting {img_path} ... {ii+1}/{len(imglist)}')
        imname = osp.basename(img_path).replace(Path(img_path).suffix, '')
        cimg = Image.open(img_path).convert('RGB')
        W, H = cimg.width, cimg.height
        H, W = long_side_to(H, W, args.long_side)
        data['width'], data['height'] = W, H
        img_resized = cimg.resize((W, H), resample=Image.Resampling.LANCZOS)


        if not osp.exists(args.save_dir):
            os.makedirs(args.save_dir)

        if args.onebyone:

            repaint_args = {
                'mask_blur': args.mask_blur,
                'inpainting_fill': INPAINTING_FILL_METHODS.index(args.fill),
                'inpaint_full_res': args.inpaint_full_res,
                'inpaint_full_res_padding': args.inpaint_full_res_padding,
                'denoising_strength': args.denoising_strength
            }
            data_inpaint = deepcopy(data)
            data_inpaint.update(repaint_args)

            from utils.io_utils import json2dict, dict2json, find_all_imgs
        
            cache_masks_dir = args.cache_masks_dir
            
            mask_fg = None
            masks = []

            if not osp.exists(cache_masks_dir):
                os.makedirs(cache_masks_dir)
            

            promptp = osp.join(cache_masks_dir, f'{imname}_prompts.json')
            bg_prompt = ''
            fg_prompts = []
            if not osp.exists(promptp):
                if detector is None:
                    from animeinsseg import AnimeInsSeg
                    import numpy as np
                    from animeinsseg.inpainting import patch_match
                    detector = AnimeInsSeg(args.detector_ckpt, device='cuda')
                    detector.init_tagger()
                instances = detector.infer(img_path, output_type='numpy', infer_tags=True)
                if not instances.is_empty:
                    prompts_dict = {}
                    for ii, mask in enumerate(instances.masks):
                        mask = cv2.resize(mask.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_AREA)
                        mask = Image.fromarray(mask)
                        savename = imname + '_' + str(ii).zfill(3) + '.png'
                        mask.save(osp.join(cache_masks_dir, savename))
                        masks.append(mask)
                        tags = instances.tags[ii].split(' ')
                        ctags = instances.character_tags[ii]
                        for ctag in ctags:
                            if ctag in tags:
                                tags.remove(ctag)
                        prompt = ','.join(tags).replace('_', ' ')
                        prompts_dict[savename] = prompt
                        fg_prompts.append(prompt)

                    mask_fg = cv2.resize(instances.compose_masks().astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_AREA)
                    bg = patch_match.inpaint(np.array(img_resized), mask_fg, patch_size=3)
                    savep = osp.join(cache_masks_dir, f'{imname}_bg_repaint.png')
                    Image.fromarray(bg).save(savep)
                    mask_fg = Image.fromarray(mask_fg)
                    mask_fg.save(osp.join(cache_masks_dir, f'{imname}_mask_fg.png'))
                    bg_tags, character_tags = detector.tagger.label_cv2_bgr(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
                    for ii, t in enumerate(bg_tags):
                        bg_tags[ii] = t.replace('_', ' ')
                    bg_prompt = ','.join(bg_tags)
                    prompts_dict[f'{imname}_bg_repaint.png'] = bg_prompt
                    dict2json(prompts_dict, promptp)
            else:
                maskp_list = find_all_imgs(cache_masks_dir, abs_path=False)
                prompts_dict = json2dict(promptp)
                for maskn in prompts_dict.keys():
                    maskp = osp.join(cache_masks_dir, maskn)
                    mask = Image.open(maskp)
                    if maskn.endswith('bg_repaint.png'):
                        bg_prompt = prompts_dict[maskn]
                        bg = mask
                    else:
                        mask = mask.convert('L')
                        fg_prompts.append(prompts_dict[maskn])
                        masks.append(mask)
                mask_fg = osp.join(cache_masks_dir, f'{imname}_mask_fg.png')
                mask_fg = Image.open(mask_fg).convert('L')

            if len(masks) == 0:
                print('no fg is found')
                continue

        for ii in tqdm(range(args.niter)):
            if args.random_seed:
                data['seed'] = randint(0, 65536)
            else:
                data['seed'] += ii
            seed = data['seed']

            if args.onebyone:
                data_inpaint['seed'] = seed
                if ii == 0:
                    img_b64 = img2b64(bg)
                    img_repainted = img_resized
                else:
                    img_b64 = output_img_b64

                if ii == 0:
                    nprompt = args.bg_nprompt
                    prompt = args.style_prompt + ','
                    if args.apply_bg_tagger:
                        prompt += bg_prompt + ','
                    prompt = prompt.strip(',')
                    
                    data['alwayson_scripts']['controlnet']['args'][0]['input_image'] = img_b64
                    data['init_images'] = [img_b64]
                    data['negative_prompt'] = nprompt
                    data['prompt'] = prompt
                    response = submit_request(args.url, json.dumps(data), auth=auth)
                    output_img_b64 = response.json()['images'][0]
                    bg_repainted = Image.open(io.BytesIO(base64.b64decode(output_img_b64)))
                    # bg_repainted.show()
                    img_repainted = Image.composite(img_repainted, bg_repainted, mask_fg)
                    # img_repainted.show()

                for jj, (fg_prompt, mask) in enumerate(zip(fg_prompts, masks)):
                    nprompt = args.global_nprompt
                    prompt = args.style_prompt + ','
                    if args.apply_fg_tagger:
                        prompt += fg_prompt + ','
                    print(prompt)
                    prompt = prompt.strip(',')

                    output_img_b64 = run_sdinpaint(img_repainted, mask, data_inpaint, prompt=prompt, nprompt=nprompt, url=args.url, auth=auth)
                    img_repainted = Image.open(io.BytesIO(base64.b64decode(output_img_b64)))
                    # img_repainted.show()

                save_encoded_image(output_img_b64, osp.join(args.save_dir, f'{imname}_onebyone_niter{ii}_output_{seed}.png'))
            else:
                img_b64 = img2b64(cimg)
                data['alwayson_scripts']['controlnet']['args'][0]['input_image'] = img_b64
                data['init_images'] = [img_b64]

                prompt = args.style_prompt + ','
                prompt = prompt.strip(',')
                data['prompt'] = prompt
                data['negative_prompt'] = args.global_nprompt

                response = submit_request(args.url, json.dumps(data), auth=auth)
                output_img_b64 = response.json()['images'][0]
                imgsavep = osp.join(args.save_dir, f'{imname}_niter{ii}_output_{seed}.png')
                save_encoded_image(output_img_b64, imgsavep)
                cimg = Image.open(io.BytesIO(base64.b64decode(output_img_b64)))