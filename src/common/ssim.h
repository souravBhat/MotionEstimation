#ifndef SSIM_H
#define SSIM_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "block.h"
#include "prediction_frame.h"

float computeMean(int *Frame, int TopLeftX, int TopLeftY, block blk, int frameWidth);
float computeVar(int *Frame, int TopLeftX, int TopLeftY, block blk, int frameWidth, float Mean);
float computeCrossVar(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk, int frameWidth, int Meanref, int Meanpred);
float computeSSIM(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk, int frameWidth);
float* findBestMatchSSIM(predictionFrame pf, int *referenceFrame, block blk, int windowTopLeftX, int windowTopLeftY, int windowBottomRightX, int windowBottomRightY);

#endif
