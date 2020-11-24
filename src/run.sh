#!/bin/bash
mkdir -p ../bin
# Run on cpu.
gcc block.c prediction_frame.c main.c utils.c -o ../bin/mes && ../bin/mes ../frames/YF1.yuv ../frames/YF2.yuv ../results/cpu/race