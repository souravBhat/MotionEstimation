#!/bin/bash
mkdir -p ../bin
nvcc block.cu prediction_frame.cu main.cu -o ../bin/mes && ../bin/mes