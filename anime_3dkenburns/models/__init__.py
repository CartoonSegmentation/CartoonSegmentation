import torch

from .disparity_refinement import Refine
from .pointcloud_inpainting import Inpaint
from .disparity_estimation import Semantics, Disparity

def load_depth_refinenet(ckpt: str, device='cpu'):
    model = Refine()
    ckpt = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt)
    return model.eval().to(device)

def disparity_refinement(net: Refine, tenImage, tenDisparity):
    return net(tenImage, tenDisparity)

def load_inpaintnet(ckpt: str, device='cpu'):
    model = Inpaint()
    ckpt = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt)
    return model.eval().to(device)

def pointcloud_inpainting(net: Inpaint, tenImage, tenDisparity, tenShift, cfg, segmask=None):
    return net(tenImage, tenDisparity, tenShift, cfg, segmask)


NETSEMANTICS = Semantics

netSemantics = None
netDisparity = None

def disparity_estimation(tenImage: torch.Tensor):

    global netSemantics
    global netDisparity

    if netSemantics is None or netDisparity is None:
        netSemantics = Semantics().eval().to(tenImage.device)
        netDisparity = Disparity().eval()
        netDisparity.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/kenburns/network-disparity.pytorch', file_name='kenburns-disparity').items() })
        netDisparity = netDisparity.to(tenImage.device)

    intWidth = tenImage.shape[3]
    intHeight = tenImage.shape[2]

    fltRatio = float(intWidth) / float(intHeight)

    intWidth = min(int(512 * fltRatio), 512)
    intHeight = min(int(512 / fltRatio), 512)

    tenImage = torch.nn.functional.interpolate(input=tenImage, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    return netDisparity(tenImage, netSemantics(tenImage))