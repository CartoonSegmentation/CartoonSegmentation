import torch
import cupy
import torchvision
import os
import re 
import os.path as osp
from utils.cupy_utils import launch_kernel, preprocess_kernel

def spatial_filter(tenInput, strType):
    tenOutput = None

    if strType == 'laplacian':
        tenLaplacian = tenInput.new_zeros(tenInput.shape[1], tenInput.shape[1], 3, 3)

        for intKernel in range(tenInput.shape[1]):
            tenLaplacian[intKernel, intKernel, 0, 1] = -1.0
            tenLaplacian[intKernel, intKernel, 0, 2] = -1.0
            tenLaplacian[intKernel, intKernel, 1, 1] = 4.0
            tenLaplacian[intKernel, intKernel, 1, 0] = -1.0
            tenLaplacian[intKernel, intKernel, 2, 0] = -1.0
        # end

        tenOutput = torch.nn.functional.pad(input=tenInput, pad=[ 1, 1, 1, 1 ], mode='replicate')
        tenOutput = torch.nn.functional.conv2d(input=tenOutput, weight=tenLaplacian)

    elif strType == 'median-3':
        tenOutput = torch.nn.functional.pad(input=tenInput, pad=[ 1, 1, 1, 1 ], mode='reflect')
        tenOutput = tenOutput.unfold(2, 3, 1).unfold(3, 3, 1)
        tenOutput = tenOutput.contiguous().view(tenOutput.shape[0], tenOutput.shape[1], tenOutput.shape[2], tenOutput.shape[3], 3 * 3)
        tenOutput = tenOutput.median(-1, False)[0]

    elif strType == 'median-5':
        tenOutput = torch.nn.functional.pad(input=tenInput, pad=[ 2, 2, 2, 2 ], mode='reflect')
        tenOutput = tenOutput.unfold(2, 5, 1).unfold(3, 5, 1)
        tenOutput = tenOutput.contiguous().view(tenOutput.shape[0], tenOutput.shape[1], tenOutput.shape[2], tenOutput.shape[3], 5 * 5)
        tenOutput = tenOutput.median(-1, False)[0]

    # end

    return tenOutput


def depth_to_points(tenDepth, fltFocal):
    tenHorizontal = torch.linspace(start=(-0.5 * tenDepth.shape[3]) + 0.5, end=(0.5 * tenDepth.shape[3]) - 0.5, steps=tenDepth.shape[3], dtype=tenDepth.dtype, device=tenDepth.device).view(1, 1, 1, -1).repeat(tenDepth.shape[0], 1, tenDepth.shape[2], 1)
    tenHorizontal = tenHorizontal * (1.0 / fltFocal)

    tenVertical = torch.linspace(start=(-0.5 * tenDepth.shape[2]) + 0.5, end=(0.5 * tenDepth.shape[2]) - 0.5, steps=tenDepth.shape[2], dtype=tenDepth.dtype, device=tenDepth.device).view(1, 1, -1, 1).repeat(tenDepth.shape[0], 1, 1, tenDepth.shape[3])
    tenVertical = tenVertical * (1.0 / fltFocal)

    return torch.cat([ tenDepth * tenHorizontal, tenDepth * tenVertical, tenDepth ], 1)





def render_pointcloud(tenInput, tenData, intWidth, intHeight, fltFocal, fltBaseline):
    tenData = torch.cat([ tenData, tenData.new_ones([ tenData.shape[0], 1, tenData.shape[2] ]) ], 1)

    tenZee = tenInput.new_zeros([ tenData.shape[0], 1, intHeight, intWidth ]).fill_(1000000.0)
    tenOutput = tenInput.new_zeros([ tenData.shape[0], tenData.shape[1], intHeight, intWidth ])

    n = tenInput.shape[0] * tenInput.shape[2]
    launch_kernel('kernel_pointrender_updateZee', preprocess_kernel('''
        extern "C" __global__ void kernel_pointrender_updateZee(
            const int n,
            const float* input,
            const float* data,
            const float* zee
        ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int intSample = ( intIndex / SIZE_2(input) ) % SIZE_0(input);
            const int intPoint  = ( intIndex                 ) % SIZE_2(input);

            assert(SIZE_1(input) == 3);
            assert(SIZE_1(zee) == 1);

            float3 fltPlanePoint = make_float3(0.0, 0.0, {{fltFocal}});
            float3 fltPlaneNormal = make_float3(0.0, 0.0, 1.0);

            float3 fltLinePoint = make_float3(VALUE_3(input, intSample, 0, intPoint), VALUE_3(input, intSample, 1, intPoint), VALUE_3(input, intSample, 2, intPoint));
            float3 fltLineVector = make_float3(0.0, 0.0, 0.0) - fltLinePoint;

            if (fltLinePoint.z < 0.001) {
                return;
            }

            float fltNumerator = dot(fltPlanePoint - fltLinePoint, fltPlaneNormal);
            float fltDenominator = dot(fltLineVector, fltPlaneNormal);
            float fltDistance = fltNumerator / fltDenominator;

            if (fabs(fltDenominator) < 0.001) {
                return;
            }

            float3 fltIntersection = fltLinePoint + (fltDistance * fltLineVector); // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

            float fltOutputX = fltIntersection.x + (0.5 * SIZE_3(zee)) - 0.5;
            float fltOutputY = fltIntersection.y + (0.5 * SIZE_2(zee)) - 0.5;

            float fltError = 1000000.0 - (({{fltFocal}} * {{fltBaseline}}) / (fltLinePoint.z + 0.0000001));

            int intNorthwestX = (int) (floor(fltOutputX));
            int intNorthwestY = (int) (floor(fltOutputY));
            int intNortheastX = intNorthwestX + 1;
            int intNortheastY = intNorthwestY;
            int intSouthwestX = intNorthwestX;
            int intSouthwestY = intNorthwestY + 1;
            int intSoutheastX = intNorthwestX + 1;
            int intSoutheastY = intNorthwestY + 1;

            float fltNorthwest = (intSoutheastX - fltOutputX)    * (intSoutheastY - fltOutputY);
            float fltNortheast = (fltOutputX    - intSouthwestX) * (intSouthwestY - fltOutputY);
            float fltSouthwest = (intNortheastX - fltOutputX)    * (fltOutputY    - intNortheastY);
            float fltSoutheast = (fltOutputX    - intNorthwestX) * (fltOutputY    - intNorthwestY);

            if ((fltNorthwest >= fltNortheast) && (fltNorthwest >= fltSouthwest) && (fltNorthwest >= fltSoutheast)) {
                if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(zee)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(zee))) {
                    atomicMin(&zee[OFFSET_4(zee, intSample, 0, intNorthwestY, intNorthwestX)], fltError);
                }

            } else if ((fltNortheast >= fltNorthwest) && (fltNortheast >= fltSouthwest) && (fltNortheast >= fltSoutheast)) {
                if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(zee)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(zee))) {
                    atomicMin(&zee[OFFSET_4(zee, intSample, 0, intNortheastY, intNortheastX)], fltError);
                }

            } else if ((fltSouthwest >= fltNorthwest) && (fltSouthwest >= fltNortheast) && (fltSouthwest >= fltSoutheast)) {
                if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(zee)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(zee))) {
                    atomicMin(&zee[OFFSET_4(zee, intSample, 0, intSouthwestY, intSouthwestX)], fltError);
                }

            } else if ((fltSoutheast >= fltNorthwest) && (fltSoutheast >= fltNortheast) && (fltSoutheast >= fltSouthwest)) {
                if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(zee)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(zee))) {
                    atomicMin(&zee[OFFSET_4(zee, intSample, 0, intSoutheastY, intSoutheastX)], fltError);
                }

            }
        } }
    ''', {
        'intWidth': intWidth,
        'intHeight': intHeight,
        'fltFocal': fltFocal,
        'fltBaseline': fltBaseline,
        'input': tenInput,
        'data': tenData,
        'zee': tenZee
    }))(
        grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
        block=tuple([ 512, 1, 1 ]),
        args=[ cupy.int32(n), tenInput.data_ptr(), tenData.data_ptr(), tenZee.data_ptr() ]
    )

    n = tenZee.nelement()
    launch_kernel('kernel_pointrender_updateDegrid', preprocess_kernel('''
        extern "C" __global__ void kernel_pointrender_updateDegrid(
            const int n,
            const float* input,
            const float* data,
            float* zee
        ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int intN = ( intIndex / SIZE_3(zee) / SIZE_2(zee) / SIZE_1(zee) ) % SIZE_0(zee);
            const int intC = ( intIndex / SIZE_3(zee) / SIZE_2(zee)               ) % SIZE_1(zee);
            const int intY = ( intIndex / SIZE_3(zee)                             ) % SIZE_2(zee);
            const int intX = ( intIndex                                           ) % SIZE_3(zee);

            assert(SIZE_1(input) == 3);
            assert(SIZE_1(zee) == 1);

            int intCount = 0;
            float fltSum = 0.0;

            int intOpposingX[] = {  1,  0,  1,  1 };
            int intOpposingY[] = {  0,  1,  1, -1 };

            for (int intOpposing = 0; intOpposing < 4; intOpposing += 1) {
                int intOneX = intX + intOpposingX[intOpposing];
                int intOneY = intY + intOpposingY[intOpposing];
                int intTwoX = intX - intOpposingX[intOpposing];
                int intTwoY = intY - intOpposingY[intOpposing];

                if ((intOneX < 0) | (intOneX >= SIZE_3(zee)) | (intOneY < 0) | (intOneY >= SIZE_2(zee))) {
                    continue;

                } else if ((intTwoX < 0) | (intTwoX >= SIZE_3(zee)) | (intTwoY < 0) | (intTwoY >= SIZE_2(zee))) {
                    continue;

                }

                if (VALUE_4(zee, intN, intC, intY, intX) >= VALUE_4(zee, intN, intC, intOneY, intOneX) + 1.0) {
                    if (VALUE_4(zee, intN, intC, intY, intX) >= VALUE_4(zee, intN, intC, intTwoY, intTwoX) + 1.0) {
                        intCount += 2;
                        fltSum += VALUE_4(zee, intN, intC, intOneY, intOneX);
                        fltSum += VALUE_4(zee, intN, intC, intTwoY, intTwoX);
                    }
                }
            }

            if (intCount > 0) {
                zee[OFFSET_4(zee, intN, intC, intY, intX)] = min(VALUE_4(zee, intN, intC, intY, intX), fltSum / intCount);
            }
        } }
    ''', {
        'intWidth': intWidth,
        'intHeight': intHeight,
        'fltFocal': fltFocal,
        'fltBaseline': fltBaseline,
        'input': tenInput,
        'data': tenData,
        'zee': tenZee
    }))(
        grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
        block=tuple([ 512, 1, 1 ]),
        args=[ cupy.int32(n), tenInput.data_ptr(), tenData.data_ptr(), tenZee.data_ptr() ]
    )

    n = tenInput.shape[0] * tenInput.shape[2]
    launch_kernel('kernel_pointrender_updateOutput', preprocess_kernel('''
        extern "C" __global__ void kernel_pointrender_updateOutput(
            const int n,
            const float* input,
            const float* data,
            const float* zee,
            float* output
        ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int intSample = ( intIndex / SIZE_2(input) ) % SIZE_0(input);
            const int intPoint  = ( intIndex                 ) % SIZE_2(input);

            assert(SIZE_1(input) == 3);
            assert(SIZE_1(zee) == 1);

            float3 fltPlanePoint = make_float3(0.0, 0.0, {{fltFocal}});
            float3 fltPlaneNormal = make_float3(0.0, 0.0, 1.0);

            float3 fltLinePoint = make_float3(VALUE_3(input, intSample, 0, intPoint), VALUE_3(input, intSample, 1, intPoint), VALUE_3(input, intSample, 2, intPoint));
            float3 fltLineVector = make_float3(0.0, 0.0, 0.0) - fltLinePoint;

            if (fltLinePoint.z < 0.001) {
                return;
            }

            float fltNumerator = dot(fltPlanePoint - fltLinePoint, fltPlaneNormal);
            float fltDenominator = dot(fltLineVector, fltPlaneNormal);
            float fltDistance = fltNumerator / fltDenominator;

            if (fabs(fltDenominator) < 0.001) {
                return;
            }

            float3 fltIntersection = fltLinePoint + (fltDistance * fltLineVector); // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

            float fltOutputX = fltIntersection.x + (0.5 * SIZE_3(output)) - 0.5;
            float fltOutputY = fltIntersection.y + (0.5 * SIZE_2(output)) - 0.5;

            float fltError = 1000000.0 - (({{fltFocal}} * {{fltBaseline}}) / (fltLinePoint.z + 0.0000001));

            int intNorthwestX = (int) (floor(fltOutputX));
            int intNorthwestY = (int) (floor(fltOutputY));
            int intNortheastX = intNorthwestX + 1;
            int intNortheastY = intNorthwestY;
            int intSouthwestX = intNorthwestX;
            int intSouthwestY = intNorthwestY + 1;
            int intSoutheastX = intNorthwestX + 1;
            int intSoutheastY = intNorthwestY + 1;

            float fltNorthwest = (intSoutheastX - fltOutputX)    * (intSoutheastY - fltOutputY);
            float fltNortheast = (fltOutputX    - intSouthwestX) * (intSouthwestY - fltOutputY);
            float fltSouthwest = (intNortheastX - fltOutputX)    * (fltOutputY    - intNortheastY);
            float fltSoutheast = (fltOutputX    - intNorthwestX) * (fltOutputY    - intNorthwestY);

            if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(output)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(output))) {
                if (fltError <= VALUE_4(zee, intSample, 0, intNorthwestY, intNorthwestX) + 1.0) {
                    for (int intData = 0; intData < SIZE_1(data); intData += 1) {
                        atomicAdd(&output[OFFSET_4(output, intSample, intData, intNorthwestY, intNorthwestX)], VALUE_3(data, intSample, intData, intPoint) * fltNorthwest);
                    }
                }
            }

            if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(output)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(output))) {
                if (fltError <= VALUE_4(zee, intSample, 0, intNortheastY, intNortheastX) + 1.0) {
                    for (int intData = 0; intData < SIZE_1(data); intData += 1) {
                        atomicAdd(&output[OFFSET_4(output, intSample, intData, intNortheastY, intNortheastX)], VALUE_3(data, intSample, intData, intPoint) * fltNortheast);
                    }
                }
            }

            if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(output)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(output))) {
                if (fltError <= VALUE_4(zee, intSample, 0, intSouthwestY, intSouthwestX) + 1.0) {
                    for (int intData = 0; intData < SIZE_1(data); intData += 1) {
                        atomicAdd(&output[OFFSET_4(output, intSample, intData, intSouthwestY, intSouthwestX)], VALUE_3(data, intSample, intData, intPoint) * fltSouthwest);
                    }
                }
            }

            if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(output)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(output))) {
                if (fltError <= VALUE_4(zee, intSample, 0, intSoutheastY, intSoutheastX) + 1.0) {
                    for (int intData = 0; intData < SIZE_1(data); intData += 1) {
                        atomicAdd(&output[OFFSET_4(output, intSample, intData, intSoutheastY, intSoutheastX)], VALUE_3(data, intSample, intData, intPoint) * fltSoutheast);
                    }
                }
            }
        } }
    ''', {
        'intWidth': intWidth,
        'intHeight': intHeight,
        'fltFocal': fltFocal,
        'fltBaseline': fltBaseline,
        'input': tenInput,
        'data': tenData,
        'zee': tenZee,
        'output': tenOutput
    }))(
        grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
        block=tuple([ 512, 1, 1 ]),
        args=[ cupy.int32(n), tenInput.data_ptr(), tenData.data_ptr(), tenZee.data_ptr(), tenOutput.data_ptr() ]
    )

    return tenOutput[:, :-1, :, :] / (tenOutput[:, -1:, :, :] + 0.0000001), tenOutput[:, -1:, :, :].detach().clone()