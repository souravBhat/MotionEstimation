#!/bin/bash

nvcc MEvar1.cu -o gpu YUVreadfile.cu YUVwritefile.cu
./gpu ./dataset/Jockey_3840x2160YF1.yuv ./dataset/Jockey_3840x2160YF2.yuv
