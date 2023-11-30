import torch
import numpy as np
import cupy

from .models.utils import spatial_filter, depth_to_points, render_pointcloud, launch_kernel, preprocess_kernel
# from .models.disparity_adjustment import disparity_adjustment
# from .models.disparity_estimation import disparity_estimation
# from .models.disparity_refinement import disparity_refinement
# from .models.pointcloud_inpainting import pointcloud_inpainting

# def process_load(npyImage, objSettings, objCommon):
#     objCommon['fltFocal'] = 1024 / 2.0
#     objCommon['fltBaseline'] = 40.0
#     objCommon['intWidth'] = npyImage.shape[1]
#     objCommon['intHeight'] = npyImage.shape[0]

#     tenImage = torch.FloatTensor(np.ascontiguousarray(npyImage.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()
#     tenDisparity = disparity_estimation(tenImage)
#     tenDisparity = disparity_adjustment(tenImage, tenDisparity)
#     tenDisparity = disparity_refinement(tenImage, tenDisparity)
#     tenDisparity = tenDisparity / tenDisparity.max() * objCommon['fltBaseline']
#     tenDepth = (objCommon['fltFocal'] * objCommon['fltBaseline']) / (tenDisparity + 0.0000001)
#     tenValid = (spatial_filter(tenDisparity / tenDisparity.max(), 'laplacian').abs() < 0.03).float()
#     tenPoints = depth_to_points(tenDepth * tenValid, objCommon['fltFocal'])
#     tenUnaltered = depth_to_points(tenDepth, objCommon['fltFocal'])

#     objCommon['fltDispmin'] = tenDisparity.min().item()
#     objCommon['fltDispmax'] = tenDisparity.max().item()
#     objCommon['objDepthrange'] = cv2.minMaxLoc(src=tenDepth[0, 0, 128:-128, 128:-128].detach().cpu().numpy(), mask=None)
#     objCommon['tenRawImage'] = tenImage
#     objCommon['tenRawDisparity'] = tenDisparity
#     objCommon['tenRawDepth'] = tenDepth
#     objCommon['tenRawPoints'] = tenPoints.view(1, 3, -1)
#     objCommon['tenRawUnaltered'] = tenUnaltered.view(1, 3, -1)

#     objCommon['tenInpaImage'] = objCommon['tenRawImage'].view(1, 3, -1)
#     objCommon['tenInpaDisparity'] = objCommon['tenRawDisparity'].view(1, 1, -1)
#     objCommon['tenInpaDepth'] = objCommon['tenRawDepth'].view(1, 1, -1)
#     objCommon['tenInpaPoints'] = objCommon['tenRawPoints'].view(1, 3, -1)
# end

# def process_inpaint(tenShift, objCommon):
#     objInpainted = pointcloud_inpainting(objCommon['tenRawImage'], objCommon['tenRawDisparity'], tenShift, objCommon)

#     objInpainted['tenDepth'] = (objCommon['fltFocal'] * objCommon['fltBaseline']) / (objInpainted['tenDisparity'] + 0.0000001)
#     objInpainted['tenValid'] = (spatial_filter(objInpainted['tenDisparity'] / objInpainted['tenDisparity'].max(), 'laplacian').abs() < 0.03).float()
#     objInpainted['tenPoints'] = depth_to_points(objInpainted['tenDepth'] * objInpainted['tenValid'], objCommon['fltFocal'])
#     objInpainted['tenPoints'] = objInpainted['tenPoints'].view(1, 3, -1)
#     objInpainted['tenPoints'] = objInpainted['tenPoints'] - tenShift

#     tenMask = (objInpainted['tenExisting'] == 0.0).view(1, 1, -1)

#     objCommon['tenInpaImage'] = torch.cat([ objCommon['tenInpaImage'], objInpainted['tenImage'].view(1, 3, -1)[tenMask.repeat(1, 3, 1)].view(1, 3, -1) ], 2)
#     objCommon['tenInpaDisparity'] = torch.cat([ objCommon['tenInpaDisparity'], objInpainted['tenDisparity'].view(1, 1, -1)[tenMask.repeat(1, 1, 1)].view(1, 1, -1) ], 2)
#     objCommon['tenInpaDepth'] = torch.cat([ objCommon['tenInpaDepth'], objInpainted['tenDepth'].view(1, 1, -1)[tenMask.repeat(1, 1, 1)].view(1, 1, -1) ], 2)
#     objCommon['tenInpaPoints'] = torch.cat([ objCommon['tenInpaPoints'], objInpainted['tenPoints'].view(1, 3, -1)[tenMask.repeat(1, 3, 1)].view(1, 3, -1) ], 2)
# end

def process_shift(objSettings, objCommon):
    fltClosestDepth = objCommon['objDepthrange'][0] + (objSettings['fltDepthTo'] - objSettings['fltDepthFrom'])
    fltClosestFromU = objCommon['objDepthrange'][2][0]
    fltClosestFromV = objCommon['objDepthrange'][2][1]
    fltClosestToU = fltClosestFromU + objSettings['fltShiftU']
    fltClosestToV = fltClosestFromV + objSettings['fltShiftV']
    fltClosestFromX = ((fltClosestFromU - (objCommon['intWidth'] / 2.0)) * fltClosestDepth) / objCommon['fltFocal']
    fltClosestFromY = ((fltClosestFromV - (objCommon['intHeight'] / 2.0)) * fltClosestDepth) / objCommon['fltFocal']
    fltClosestToX = ((fltClosestToU - (objCommon['intWidth'] / 2.0)) * fltClosestDepth) / objCommon['fltFocal']
    fltClosestToY = ((fltClosestToV - (objCommon['intHeight'] / 2.0)) * fltClosestDepth) / objCommon['fltFocal']

    fltShiftX = fltClosestFromX - fltClosestToX
    fltShiftY = fltClosestFromY - fltClosestToY
    fltShiftZ = objSettings['fltDepthTo'] - objSettings['fltDepthFrom']

    tenShift = torch.FloatTensor([ fltShiftX, fltShiftY, fltShiftZ ]).view(1, 3, 1).cuda()

    tenPoints = objSettings['tenPoints'].clone()

    tenPoints[:, 0:1, :] *= tenPoints[:, 2:3, :] / (objSettings['tenPoints'][:, 2:3, :] + 0.0000001)
    tenPoints[:, 1:2, :] *= tenPoints[:, 2:3, :] / (objSettings['tenPoints'][:, 2:3, :] + 0.0000001)

    tenPoints += tenShift

    return tenPoints, tenShift
# end

def process_autozoom(objSettings, objCommon):
    npyShiftU = np.linspace(-objSettings['fltShift'], objSettings['fltShift'], 16)[None, :].repeat(16, 0)
    npyShiftV = np.linspace(-objSettings['fltShift'], objSettings['fltShift'], 16)[:, None].repeat(16, 1)
    fltCropWidth = objSettings['objFrom']['intCropWidth'] / objSettings['fltZoom']
    fltCropHeight = objSettings['objFrom']['intCropHeight'] / objSettings['fltZoom']

    fltDepthFrom = objCommon['objDepthrange'][0]
    fltDepthTo = objCommon['objDepthrange'][0] * (fltCropWidth / objSettings['objFrom']['intCropWidth'])

    fltBest = 0.0
    fltBestU = None
    fltBestV = None

    for intU in range(16):
        for intV in range(16):
            fltShiftU = npyShiftU[intU, intV].item()
            fltShiftV = npyShiftV[intU, intV].item()

            if objSettings['objFrom']['fltCenterU'] + fltShiftU < fltCropWidth / 2.0:
                continue

            elif objSettings['objFrom']['fltCenterU'] + fltShiftU > objCommon['intWidth'] - (fltCropWidth / 2.0):
                continue

            elif objSettings['objFrom']['fltCenterV'] + fltShiftV < fltCropHeight / 2.0:
                continue

            elif objSettings['objFrom']['fltCenterV'] + fltShiftV > objCommon['intHeight'] - (fltCropHeight / 2.0):
                continue

            # end

            tenPoints = process_shift({
                'tenPoints': objCommon['tenRawPoints'],
                'fltShiftU': fltShiftU,
                'fltShiftV': fltShiftV,
                'fltDepthFrom': fltDepthFrom,
                'fltDepthTo': fltDepthTo
            }, objCommon)[0]

            tenRender, tenExisting = render_pointcloud(tenPoints, objCommon['tenRawImage'].view(1, 3, -1), objCommon['intWidth'], objCommon['intHeight'], objCommon['fltFocal'], objCommon['fltBaseline'])

            if fltBest < (tenExisting > 0.0).float().sum().item():
                fltBest = (tenExisting > 0.0).float().sum().item()
                fltBestU = fltShiftU
                fltBestV = fltShiftV
            # end
        # end
    # end

    return {
        'fltCenterU': objSettings['objFrom']['fltCenterU'] + fltBestU,
        'fltCenterV': objSettings['objFrom']['fltCenterV'] + fltBestV,
        'intCropWidth': int(round(objSettings['objFrom']['intCropWidth'] / objSettings['fltZoom'])),
        'intCropHeight': int(round(objSettings['objFrom']['intCropHeight'] / objSettings['fltZoom']))
    }
# end


def fill_disocclusion(tenInput, tenDepth):
    tenOutput = tenInput.clone()

    n = tenInput.shape[0] * tenInput.shape[2] * tenInput.shape[3]
    launch_kernel('kernel_discfill_updateOutput', preprocess_kernel('''
        extern "C" __global__ void kernel_discfill_updateOutput(
            const int n,
            const float* input,
            const float* depth,
            float* output
        ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int intSample = ( intIndex / SIZE_3(input) / SIZE_2(input) ) % SIZE_0(input);
            const int intY      = ( intIndex / SIZE_3(input)                 ) % SIZE_2(input);
            const int intX      = ( intIndex                                 ) % SIZE_3(input);

            assert(SIZE_1(depth) == 1);

            if (VALUE_4(depth, intSample, 0, intY, intX) > 0.0) {
                return;
            }

            float fltShortest = 1000000.0;

            int intFillX = -1;
            int intFillY = -1;

            float fltDirectionX[] = { -1, 0, 1, 1,    -1, 1, 2,  2,    -2, -1, 1, 2, 3, 3,  3,  3 };
            float fltDirectionY[] = {  1, 1, 1, 0,     2, 2, 1, -1,     3,  3, 3, 3, 2, 1, -1, -2 };

            for (int intDirection = 0; intDirection < 16; intDirection += 1) {
                float fltNormalize = sqrt((fltDirectionX[intDirection] * fltDirectionX[intDirection]) + (fltDirectionY[intDirection] * fltDirectionY[intDirection]));

                fltDirectionX[intDirection] /= fltNormalize;
                fltDirectionY[intDirection] /= fltNormalize;
            }

            for (int intDirection = 0; intDirection < 16; intDirection += 1) {
                float fltFromX = intX; int intFromX = 0;
                float fltFromY = intY; int intFromY = 0;

                float fltToX = intX; int intToX = 0;
                float fltToY = intY; int intToY = 0;

                do {
                    fltFromX -= fltDirectionX[intDirection]; intFromX = (int) (round(fltFromX));
                    fltFromY -= fltDirectionY[intDirection]; intFromY = (int) (round(fltFromY));

                    if ((intFromX < 0) | (intFromX >= SIZE_3(input))) { break; }
                    if ((intFromY < 0) | (intFromY >= SIZE_2(input))) { break; }
                    if (VALUE_4(depth, intSample, 0, intFromY, intFromX) > 0.0) { break; }
                } while (true);
                if ((intFromX < 0) | (intFromX >= SIZE_3(input))) { continue; }
                if ((intFromY < 0) | (intFromY >= SIZE_2(input))) { continue; }

                do {
                    fltToX += fltDirectionX[intDirection]; intToX = (int) (round(fltToX));
                    fltToY += fltDirectionY[intDirection]; intToY = (int) (round(fltToY));

                    if ((intToX < 0) | (intToX >= SIZE_3(input))) { break; }
                    if ((intToY < 0) | (intToY >= SIZE_2(input))) { break; }
                    if (VALUE_4(depth, intSample, 0, intToY, intToX) > 0.0) { break; }
                } while (true);
                if ((intToX < 0) | (intToX >= SIZE_3(input))) { continue; }
                if ((intToY < 0) | (intToY >= SIZE_2(input))) { continue; }

                float fltDistance = sqrt(powf(intToX - intFromX, 2) + powf(intToY - intFromY, 2));

                if (fltShortest > fltDistance) {
                    intFillX = intFromX;
                    intFillY = intFromY;

                    if (VALUE_4(depth, intSample, 0, intFromY, intFromX) < VALUE_4(depth, intSample, 0, intToY, intToX)) {
                        intFillX = intToX;
                        intFillY = intToY;
                    }

                    fltShortest = fltDistance;
                }
            }

            if (intFillX == -1) {
                return;

            } else if (intFillY == -1) {
                return;

            }

            for (int intDepth = 0; intDepth < SIZE_1(input); intDepth += 1) {
                output[OFFSET_4(output, intSample, intDepth, intY, intX)] = VALUE_4(input, intSample, intDepth, intFillY, intFillX);
            }
        } }
    ''', {
        'input': tenInput,
        'depth': tenDepth,
        'output': tenOutput
    }))(
        grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
        block=tuple([ 512, 1, 1 ]),
        args=[ cupy.int32(n), tenInput.data_ptr(), tenDepth.data_ptr(), tenOutput.data_ptr() ]
    )

    return tenOutput
# end