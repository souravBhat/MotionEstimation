#include <stdio.h>
#include <math.h>
#include "block.h"
#include "prediction_frame.h"
#include "utils.h"

void populateBlkMotionVector(block *blk, int dx, int dy) {
  blk->motion_vectorX = dx;
  blk->motion_vectorY = dy;
  blk->is_best_match_found = 1;
}

// Given candidate block and current frame block, compute mse.
float computeMse(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk){
  float sum = 0;
  for(int offsetY = 0; offsetY < blk.height; offsetY++) {
    for(int offsetX = 0; offsetX < blk.width; offsetX++) {
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

  for(int y = windowTopLeftY; y <= windowBottomRightY - blk.height + 1; y++) {
    for(int x = windowTopLeftX; x <= windowBottomRightX - blk.width + 1; x++) {
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
float findBestBlkMse(predictionFrame pf, int *referenceFrame, block *blk, int extraSpan) {
  // Get bounds of search window.
  int topLeftX = blk->top_left_x;
  int topLeftY = blk->top_left_y;
  int bottomRightX = blk->bottom_right_x;
  int bottomRightY = blk->bottom_right_y;
  int windowTopLeftX = (topLeftX - extraSpan) < 0 ? 0 : topLeftX - extraSpan;
  int windowTopLeftY = (topLeftY - extraSpan) < 0 ? 0 : topLeftY - extraSpan;
  int windowBottomRightX = (bottomRightX + extraSpan) >= pf.width ? pf.width - 1 : (bottomRightX + extraSpan);
  int windowBottomRightY = (bottomRightY + extraSpan) >= pf.height ? pf.height - 1: (bottomRightY + extraSpan);

  float* match = findBestMatchMse(pf, referenceFrame, *blk, windowTopLeftX, windowTopLeftY, windowBottomRightX, windowBottomRightY);
  populateBlkMotionVector(blk, (int) match[1], (int)match[2]);

  return match[0];
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
      printf("Error: wrong number of argument\n");
      exit(0);
  }
  int blkDim = 16;
  int extraSpan = 15;
  int frameWidth =  3840;
  int frameHeight = 2160;
  int numElems = frameWidth * frameHeight;
  int bytes = sizeof(int) * numElems;

  printf("Current frame file name = %s\nReference frame file name = %s\n", argv[1], argv[2]);
  int * currentFrame = (int*) malloc(bytes);
  int * refFrame = (int*) malloc(bytes);
  if(!yuvReadFrame(argv[1], currentFrame, numElems)) {
    exit(1);
  };
  if(!yuvReadFrame(argv[2], refFrame, numElems)) {
    exit(1);
  }

  predictionFrame p;
  createPredictionFrame(&p, currentFrame, frameWidth, frameHeight, blkDim);
  for(int i = 0; i < p.num_blks; i++) {
    findBestBlkMse(p, refFrame, &p.blks[i], extraSpan);

    #ifdef DEBUG
    #if (DEBUG > 0)
    printf("Blk %s, score: %.3f, mv: [%d, %d]\n", blkStr(p.blks[i]), val, p.blks[i].motion_vectorX, p.blks[i].motion_vectorY);
    #endif
    #endif
  }

  int* motionCompFrame = motionCompensatedFrame(p, refFrame);
  int* motionCompDiff = (int*) malloc(sizeof(int) * numElems);
  int* originalDiff = (int*) malloc(sizeof(int) * numElems);
  frameDiff(motionCompDiff, motionCompFrame, currentFrame, numElems);
  frameDiff(originalDiff, refFrame, currentFrame, numElems);

  float motionCompScore = 0.0;
  float originalScore = 0.0;
  for(int i =0; i < numElems; i++) {
    motionCompScore += (motionCompFrame[i] - currentFrame[i]) * (motionCompFrame[i] - currentFrame[i]);
    originalScore += (currentFrame[i] - refFrame[i]) * (currentFrame[i] - refFrame[i]);
  }
  printf("Original Score: %.4f, Compensated Score: %.4f\n", originalScore/numElems, motionCompScore/numElems);
  yuvWriteFrame("../bin/motion_comp.yuv", motionCompFrame, numElems);
  yuvWriteFrame("../bin/motion_comp_diff.yuv", motionCompDiff, numElems);
  yuvWriteFrame("../bin/original_diff.yuv", originalDiff, numElems);

  for(int i = 0; i < 10; i++) {
    printf("Blk %s, mv: [%d, %d]\n", blkStr(p.blks[i]), p.blks[i].motion_vectorX, p.blks[i].motion_vectorY);
  }
}
