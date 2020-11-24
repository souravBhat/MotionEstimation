#!/bin/bash
mkdir -p ../../bin
# Run on cpu.
gcc ../common/block.c ../common/prediction_frame.c ../common/utils.c main.c -o ../../bin/mes && ../../bin/mes ../../frames/JockeyYF1.yuv ../../frames/JockeyYF2.yuv ../../results/cpu/jockey