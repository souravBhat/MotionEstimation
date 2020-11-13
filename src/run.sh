#!/bin/bash
mkdir -p ../bin
nvcc block.c prediction_frame.c main.c -o ../bin/mes && ../bin/mes