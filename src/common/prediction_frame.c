#include "prediction_frame.h"

void createPredictionFrame(predictionFrame *pf, int *frame, int width, int height, int blkDim){
  pf->frame = frame;
  pf->width = width;
  pf->height = height;
  pf->blk_dim = blkDim;

  int numBlocksX = (width + blkDim - 1)/blkDim;
  int numBlocksY = (height + blkDim - 1)/blkDim;
  int numBlocks = numBlocksX * numBlocksY;
  pf->num_blks = numBlocks;
  pf->blks = (block*)malloc(sizeof(block) * numBlocks);

  for(int i = 0; i < numBlocks; i++) {
    int blockX = i % numBlocksX;
    int blockY = i / numBlocksX;

    int topLeftX = blockX * blkDim;
    int topLeftY = blockY * blkDim;
    int blkWidth = (topLeftX + blkDim) < width ? blkDim : width - topLeftX;
    int blkHeight = (topLeftY + blkDim) < height ? blkDim: height - topLeftY;
    createBlk(&pf->blks[i], blockX, blockY, topLeftX, topLeftY, blkWidth, blkHeight);
  }
}

char* predictionFrameStr(predictionFrame pf) {
  char *c = (char*) malloc(sizeof(char) * pf.num_blks * 30 + 80);
  sprintf(c, "PredictionFrame [width=%d, height=%d, numBlocks=%d, blockDim=%d]\n",
    pf.width, pf.height, pf.num_blks, pf.blk_dim);

  char *blkC = (char*) malloc(sizeof(char)*50);
  for(int i = 0; i < pf.num_blks - 1; i++) {
    sprintf(blkC, "   %s\n", blkStr(pf.blks[i]));
    strcat(c, blkC);
  }
  sprintf(blkC, "   %s\n", blkStr(pf.blks[pf.num_blks - 1]));
  strcat(c, blkC);
  return c;
}
