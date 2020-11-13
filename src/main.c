#include <stdio.h>
#include <math.h>
#include "block.h"
#include "prediction_frame.h"

void populateBlkMotionVector(block *blk, int dx, int dy) {
  blk->motion_vector = (int*) malloc(sizeof(int) * 2);
  blk->motion_vector[0] = dx;
  blk->motion_vector[1] = dy;
  blk->is_best_match_found = 1;
}

// Given candidate block and current frame block, compute mse.
float computeMse(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk){
  float sum = 0;
  for(int offsetY = 0; offsetY < blk.height; offsetY++) {
    for(int offsetX = 0; offsetX <= blk.width; offsetX++) {
      int idxCand = (candBlkTopLeftY + offsetY) * blk.width + (candBlkTopLeftX + offsetX);
      int idxRefBlk = (blk.top_left_y + offsetY) * blk.width + (blk.top_left_x + offsetX);
      sum+= (predictionFrame[idxRefBlk] - referenceFrame[idxCand])*(predictionFrame[idxRefBlk] - referenceFrame[idxCand]);
    }
  }
  return sum/(blk.width * blk.height);
}

// Find best block using mse metric within the search window.
float* findBestMatchMse(predictionFrame pf, int *referenceFrame, block blk, int windowTopLeftX, int windowTopLeftY,
  int windowBottomRightX, int windowBottomRightY) {
  // First is the score, 2nd and 3rd element is the vector.
  float* result = (float*) malloc(sizeof(float) * 3);
  result[0] = INFINITY;
  for(int y = windowTopLeftY; y < windowBottomRightY - blk.height; y++) {
    for(int x = windowTopLeftX; x < windowBottomRightX - blk.width; x++) {
      int candBlkTopLeftX = x;
      int candBlkTopLeftY = y;
      float candMse = computeMse(referenceFrame, candBlkTopLeftX, candBlkTopLeftY, pf.frame, blk);
      if(candMse < result[0]) {
        result[0] = candMse;
        result[1] = blk.top_left_x - candBlkTopLeftX;
        result[2] = blk.top_left_y - candBlkTopLeftY;
      }
    }
  }
  return result;
}

// Returns the score of the best block, given the window size.
float findBestBlkMse(predictionFrame pf, int *referenceFrame, block *blk, int windowSize) {
  // Get bounds of search window.
  int blkCenterX = (blk->top_left_x + blk->width + 1)/2;
  int blkCenterY = (blk->top_left_y + blk->height + 1)/2;
  int windowTopLeftX = (blkCenterX - windowSize) < 0 ? 0 : blkCenterX - windowSize;
  int windowTopLeftY = (blkCenterY - windowSize) < 0 ? 0 : blkCenterY - windowSize;
  int windowBottomRightX = (blkCenterX + windowSize) >= pf.width ? pf.width - 1 : (blkCenterX + windowSize);
  int windowBottomRightY = (blkCenterY + windowSize) >= pf.height ? pf.height - 1: (blkCenterY + windowSize);

  float* match = findBestMatchMse(pf, referenceFrame, *blk, windowTopLeftX, windowTopLeftY, windowBottomRightX, windowBottomRightY);
  populateBlkMotionVector(blk, (int) match[1], (int)match[2]);

  return match[0];
}

int main() {
  block b;
  createBlk(&b, 1, 1, 2, 2, 3, 4);
  printf("%s\n", blkStr(b));

  int nx = 3; int ny = 5;
  int *frame =(int*) malloc(sizeof(int) * nx * ny);
  for(int i = 0; i < ny; i++) {
    for(int j = 0; j < nx; j++) {
      frame[i *  nx + j] = 1;
    }
  }

  predictionFrame p;
  createPredictionFrame(&p, frame, 5, 3, 2);
  printf("%s\n", predictionFrameStr(p));
  int windowSize = 3;
  float sum = 0.0;
  for(int i = 0; i < p.num_blks; i++) {
    float val = findBestBlkMse(p, frame, &p.blks[i], windowSize);
    int* mv = p.blks[i].motion_vector;
    printf("Blk %s, score: %.3f, mv: [%d, %d]\n", blkStr(p.blks[i]), val, mv[0], mv[1]);
    sum += val;
  }
  printf("%.3f\n", sum);
}