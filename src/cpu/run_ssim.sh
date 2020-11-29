#!/bin/bash
mkdir -p ../../bin
# Run on cpu.
gcc ../common/block.c ../common/prediction_frame.c ../common/ssim.c -lm ../common/utils.c main.c -o ../../bin/mes && ../../bin/mes ../../frames/ForemanYF4.yuv ../../frames/ForemanYF1.yuv ../../results/cpu/foreman 4 15 352 288
