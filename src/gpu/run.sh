#!/bin/bash
mkdir -p ../../bin

nvcc YUVreadfile.cu YUVwritefile.cu ../common/block.c ../common/prediction_frame.c GPUBaseline.cu -o ../../bin/gpu
../../bin/gpu ../../frames/JockeyYF1.yuv ../../frames/JockeyYF2.yuv
