#include "ssim.h"

float computeMean(int *Frame, int TopLeftX, int TopLeftY, block blk, int frameWidth)
{
  float sum = 0;
  for(int offsetY = 0; offsetY < blk.height; offsetY++) {
    for(int offsetX = 0; offsetX < blk.width; offsetX++) {
      int idx = (TopLeftY + offsetY) * frameWidth + (TopLeftX + offsetX);
      sum+= Frame[idx];
    }
  }
  sum = sum / (blk.width * blk.height);
  return sum;
}

float computeVar(int *Frame, int TopLeftX, int TopLeftY, block blk, int frameWidth, float Mean)
{
  float sum = 0;
  for(int offsetY = 0; offsetY < blk.height; offsetY++) {
    for(int offsetX = 0; offsetX < blk.width; offsetX++) {
      int idx = (TopLeftY + offsetY) * frameWidth + (TopLeftX + offsetX);
      sum+= (Frame[idx]-Mean)*(Frame[idx]-Mean);
    }
  }
  sum = sum / (blk.width * blk.height);
  return sum;
}

float computeCrossVar(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk, int frameWidth, int Meanref, int Meanpred)
{
  float sum = 0;
  for(int offsetY = 0; offsetY < blk.height; offsetY++) {
    for(int offsetX = 0; offsetX < blk.width; offsetX++) {
      int idxCand = (candBlkTopLeftY + offsetY) * frameWidth + (candBlkTopLeftX + offsetX);
      int idxRefBlk = (blk.top_left_y + offsetY) * frameWidth + (blk.top_left_x + offsetX);
      sum+= (referenceFrame[idxCand] - Meanref)*(predictionFrame[idxRefBlk] - Meanpred);
    }
  }
  sum = sum / (blk.width * blk.height);
  return sum;
}

// Given candidate block and current frame block, compute mse.
float computeSSIM(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk, int frameWidth){
  float Meanref, Meanpred, Varref, Varpred, StdDevref, StdDevpred, CrossVar, Luminence, Contrast, Structure, score;
  //Increasing values of constants gives lower compensated score
  float C1 = 0.01; float C2 = 0.09; float C3 = 0.045;
  Meanref = computeMean(referenceFrame, candBlkTopLeftX, candBlkTopLeftY, blk, frameWidth);
  Meanpred = computeMean(predictionFrame, blk.top_left_x, blk.top_left_y, blk, frameWidth);
  Varref = computeVar(referenceFrame, candBlkTopLeftX, candBlkTopLeftY, blk, frameWidth, Meanref);
  Varpred = computeVar(predictionFrame, blk.top_left_x, blk.top_left_y, blk, frameWidth, Meanpred);
  StdDevref = sqrt(Varref);
  StdDevpred = sqrt(Varpred);
  CrossVar = computeCrossVar(referenceFrame, candBlkTopLeftX, candBlkTopLeftY, predictionFrame, blk, frameWidth, Meanref, Meanpred);
  Luminence = (2*Meanref*Meanpred + C1)/(Meanref*Meanref + Meanpred*Meanpred + C1);
  Contrast = (2*StdDevref*StdDevpred + C2)/(StdDevref*StdDevref + StdDevpred*StdDevpred + C2);
  Structure = (CrossVar + C3)/(StdDevref*StdDevpred + C3);
  score = Luminence*Contrast*Structure;
  return score;
}

/*float computeMse(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk, int frameWidth){
  float sum = 0;
  for(int offsetY = 0; offsetY < blk.height; offsetY++) {
    for(int offsetX = 0; offsetX < blk.width; offsetX++) {
      int idxCand = (candBlkTopLeftY + offsetY) * frameWidth + (candBlkTopLeftX + offsetX);
      int idxRefBlk = (blk.top_left_y + offsetY) * frameWidth + (blk.top_left_x + offsetX);
      sum+= (predictionFrame[idxRefBlk] - referenceFrame[idxCand])*(predictionFrame[idxRefBlk] - referenceFrame[idxCand]);
    }
  }
  float score = sum / (blk.width * blk.height);
  #ifdef DEBUG
  #if (DEBUG > 0)
  if(blk.idx_x == 17 && blk.idx_y == 2) {
    printf("Search (%d, %d) -> (%d, %d). Score: %.3f\n", candBlkTopLeftX, candBlkTopLeftY, candBlkTopLeftX + blk.width - 1, candBlkTopLeftY + blk.height - 1, score);
  }
  #endif
  #endif
  return score;
}*/

// Find best block using mse metric within the search window.
float* findBestMatchSSIM(predictionFrame pf, int *referenceFrame, block blk, int windowTopLeftX, int windowTopLeftY,
  int windowBottomRightX, int windowBottomRightY) {
  // First is the score, 2nd and 3rd element is the vector.
  float* result = (float*) malloc(sizeof(float) * 3);
  result[0] = 0;

  #ifdef DEBUG
  #if (DEBUG > 0)
  if (blk.idx_x == 17 && blk.idx_y == 2) {
    printf("Blk %s, Search window: (%d, %d) -> (%d, %d)\n\n\n", blkStr(blk), windowTopLeftX, windowTopLeftY, windowBottomRightX, windowBottomRightY);
  }
  #endif
  #endif

  for(int y = windowTopLeftY; y <= windowBottomRightY - blk.height + 1; y++) {
    for(int x = windowTopLeftX; x <= windowBottomRightX - blk.width + 1; x++) {
      float candSSIM = computeSSIM(referenceFrame, x, y, pf.frame, blk,  pf.width);
      if(candSSIM > result[0]) {
        result[0] = candSSIM;
        result[1] = x - blk.top_left_x;
        result[2] = y - blk.top_left_y;
      }
    }
  }
  return result;
}
