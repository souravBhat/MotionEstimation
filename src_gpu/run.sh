#!/bin/bash

nvcc GPUBaseline.cu -o gpu YUVreadfile.cu YUVwritefile.cu ./../src/block.cu ./../src/prediction_frame.cu
./gpu ./dataset/Jockey_3840x2160YF1.yuv ./dataset/Jockey_3840x2160YF2.yuv
