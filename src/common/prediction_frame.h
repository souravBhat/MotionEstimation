#ifndef PREDICTION_FRAME_H
#define PREDICTION_FRAME_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "block.h"

typedef struct predictionFrame{
  int *frame;
  int width;
  int height;
  int blk_dim;
  int num_blks;
  block* blks;

} predictionFrame;

void createPredictionFrame(predictionFrame* pf, int *frame, int width, int height, int blockDim);
char* predictionFrameStr(predictionFrame pf);
#endif