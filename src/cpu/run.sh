#!/bin/bash
mkdir -p ../../bin
# Run on cpu.
gcc ../common/block.c ../common/prediction_frame.c ../common/utils.c main.c ./thpool.c -o ../../bin/mes -lm -lpthread && ../../bin/mes ../../frames/ForemanYF4.yuv ../../frames/ForemanYF1.yuv ../../results/cpu/foreman 
