import cv2
import moviepy
import moviepy.editor
import os
import torch
from common import process_inpaint, spatial_filter, depth_to_points, process_kenburns, disparity_estimation, disparity_adjustment, disparity_refinement
import numpy as np
import os.path as osp
import os
from utils import load_zoe, colorize
from PIL import Image

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

objCommon = {}
objPlayback = {
    'strImage': None,
    'npyImage': None,
    'strMode': 'automatic',
    'intTime': 0,
    'fltTime': np.linspace(0.0, 1.0, 75).tolist() + list(reversed(np.linspace(0.0, 1.0, 75).tolist())),
    'strCache': {},
    'objFrom': {
        'fltCenterU': 512.0,
        'fltCenterV': 384.0,
        'intCropWidth': 1024,
        'intCropHeight': 768
    },
    'objTo': {
        'fltCenterU': 512.0,
        'fltCenterV': 384.0,
        'intCropWidth': 1024,
        'intCropHeight': 768
    }
}

def process_load(img_path: str, objCommon):

    crop_ratio = 0.8

    npyImage = cv2.imread(img_path)
    im_h, im_w = npyImage.shape[:2]

    intCropWidth = int(im_w * crop_ratio)
    intCropHeight = int(im_h * crop_ratio)

    image = Image.open(img_path).convert("RGB")

    depth_model = load_zoe(r'E:\gitclones\ZoeDepth\ZoeD_M12_N.pt', device='cuda')
    # depth = depth_model.infer_pil(image, output_type='tensor', with_flip_aug=False)

    # image = Image.open(r'E:\gitclones\MiDaS\input\diff\02.png').convert("RGB")
    depth = depth_model.infer_pil(image, output_type='tensor')

    # d = 8
    d = 24
    disparity = (1 / depth) * d
    disp =(disparity / disparity.max() * 255).cpu().numpy().astype(np.uint8)
    Image.fromarray(disp).save('disp.png')
    dzoe = colorize(disparity.cpu().numpy())
    Image.fromarray(dzoe).save('dzoe.png')

    disparity = disparity.unsqueeze(0).unsqueeze(0).to('cuda')
    print(disparity.max())

    objCommon['fltFocal'] = 1024 / 2.0
    objCommon['fltBaseline'] = 40.0
    objCommon['intWidth'] = npyImage.shape[1]
    objCommon['intHeight'] = npyImage.shape[0]

    tenImage = torch.FloatTensor(np.ascontiguousarray(npyImage.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()
    tenDisparity = disparity_estimation(tenImage)
    print(tenDisparity.max())

    tenDisparity = torch.nn.functional.interpolate(input=disparity, size=(tenDisparity.shape[2], tenDisparity.shape[3]))

    t_est = tenDisparity.cpu().numpy().squeeze()
    t_est = colorize(t_est)
    Image.fromarray(t_est).save('t_est.png')

    tenDisparity = disparity_adjustment(tenImage, tenDisparity)
    t_adj = tenDisparity.cpu().numpy().squeeze()
    t_adj = colorize(t_adj)
    Image.fromarray(t_adj).save('t_adj.png')
    
    tenDisparity = disparity_refinement(tenImage, tenDisparity)
    t_ref = tenDisparity.cpu().numpy().squeeze()
    t_ref = colorize(t_ref)
    Image.fromarray(t_ref).save('t_ref.png')
    
    tenDisparity = disparity
    tenDisparity = tenDisparity / tenDisparity.max() * objCommon['fltBaseline']
    tenDepth = (objCommon['fltFocal'] * objCommon['fltBaseline']) / (tenDisparity + 0.0000001)
    tenValid = (spatial_filter(tenDisparity / tenDisparity.max(), 'laplacian').abs() < 0.03).float()
    tenPoints = depth_to_points(tenDepth * tenValid, objCommon['fltFocal'])
    tenUnaltered = depth_to_points(tenDepth, objCommon['fltFocal'])

    objCommon['fltDispmin'] = tenDisparity.min().item()
    objCommon['fltDispmax'] = tenDisparity.max().item()
    objCommon['objDepthrange'] = cv2.minMaxLoc(src=tenDepth[0, 0, 128:-128, 128:-128].detach().cpu().numpy(), mask=None)
    objCommon['tenRawImage'] = tenImage
    objCommon['tenRawDisparity'] = tenDisparity
    objCommon['tenRawDepth'] = tenDepth
    objCommon['tenRawPoints'] = tenPoints.view(1, 3, -1)
    objCommon['tenRawUnaltered'] = tenUnaltered.view(1, 3, -1)

    objCommon['tenInpaImage'] = objCommon['tenRawImage'].view(1, 3, -1)
    objCommon['tenInpaDisparity'] = objCommon['tenRawDisparity'].view(1, 1, -1)
    objCommon['tenInpaDepth'] = objCommon['tenRawDepth'].view(1, 1, -1)
    objCommon['tenInpaPoints'] = objCommon['tenRawPoints'].view(1, 3, -1)

    for fltX, fltY in [ (100.0, 0.0), (-100.0, 0.0), (0.0, 100.0), (0.0, -100.0) ]:
        process_inpaint(torch.FloatTensor([ fltX, fltY, 0.0 ]).view(1, 3, 1).cuda(), objCommon)

    # objPlayback['objFrom'] = {'fltCenterU': 482.55556640625, 'fltCenterV': 361.9166748046875, 'intCropWidth': 841, 'intCropHeight': 631} 
    # objPlayback['objTo'] = {'fltCenterU': 512.0, 'fltCenterV': 384.0, 'intCropWidth': 900, 'intCropHeight': 675}
    objPlayback['objFrom'] = {'fltCenterU': intCropWidth / 2, 'fltCenterV': intCropHeight / 2, 'intCropWidth': intCropWidth, 'intCropHeight': intCropHeight} 
    objPlayback['objTo'] = {'fltCenterU': intCropWidth / 2 + im_w - intCropWidth, 'fltCenterV': intCropHeight / 2 + im_h - intCropHeight, 'intCropWidth': intCropWidth, 'intCropHeight': intCropHeight}

    # Debug by Francis
    npyKenburns,_ = process_kenburns({
        'fltSteps': np.linspace(0.0, 1.0, 75).tolist(),
        'objFrom': objPlayback['objFrom'],
        'objTo': objPlayback['objTo'],
        'boolInpaint': True
    }, objCommon)
    return npyKenburns


if __name__ == '__main__':
    img_path = r'images\doublestrike.jpg'
    img_path = r'E:\gitclones\MiDaS\input\original\02.jpg'
    npyKenburns = process_load(img_path, objCommon)


    save_dir = 'result'
    save_name = 'kenburns.mp4'
    os.makedirs(name=save_dir, exist_ok=True)

    moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyKenburns + list(reversed(npyKenburns))[1:-1] ], fps=25).write_videofile(osp.join(save_dir, save_name))