#!/bin/bash
mkdir -p ../bin
nvcc block.cu prediction_frame.cu main.cu utils.cu -o ../bin/mes && ../bin/mes