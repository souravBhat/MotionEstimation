#include <stdio.h>
#include <math.h>
#include "block.h"
#include "prediction_frame.h"
#include "utils.h"

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

int* motionCompensatedFrame(predictionFrame pf) {
  int* motionCompFrame = (int*) malloc(sizeof(int) * pf.width * pf.height);
  for(int i = 0; i < pf.num_blks; i++) {
    block blk = pf.blks[i];
    if(!blk.is_best_match_found) continue;
    int * mv = blk.motion_vector;

    // Populate the compensated frame.
    for(int x = blk.top_left_x; x <= blk.bottom_right_x; x++) {
      for(int y = blk.top_left_y; y <= blk.bottom_right_y; y++) {
        // Get compensated position.
        int comp_x = x - mv[0];
        int comp_y = y - mv[1];
        // Populate array if within bounds.
        if(comp_x >= 0 && comp_y >= 0) {
          #ifdef DEBUG
          #if (DEBUG > 0)
          printf("(%d, %d) moved to (%d, %d)\n", x, y, comp_x, comp_y);
          #endif
          #endif
          motionCompFrame[comp_y * pf.width + comp_x] = pf.frame[y * pf.width + x];
        }
      }
    }
  }
  return motionCompFrame;
}

int main() {
  int nx = 10; int ny = 5;
  int blkDim = 1;
  int extraSpan = 1;

  int frameA[ny][nx];
  int frameB[ny][nx];
  int count = 1;

  for(int y = 0; y < ny; y++) {
      for(int x = 0; x < nx; x++) {
        frameA[y][x] = 0;
        frameB[y][x] = 0;
      }
  }

  for(int y = 1; y < ny; y++) {
    for(int x = 1; x < nx; x++) {
      frameA[y - 1][x - 1] = count;
      frameB[y][x] = count++;
    }
  }

  // Print out frame A and B. We to get motion vectors from frameA -> frameB.
  printf("FrameA:\n");
  for(int y = 0; y < ny; y++) {
      for(int x = 0; x < nx; x++) {
        printf("%d\t", frameA[y][x]);
      }
      printf("\n");
  }

  printf("\nFrameB:\n");
  for(int y = 0; y < ny; y++) {
      for(int x = 0; x < nx; x++) {
        printf("%d\t", frameB[y][x]);
      }
      printf("\n");
  }

  int *frameAPtr = (int*) malloc(sizeof(int) * nx * ny);
  int *frameBPtr = (int*) malloc(sizeof(int) * nx * ny);
  for(int i = 0; i < ny; i++) {
    for(int j = 0; j < nx; j++) {
      frameAPtr[i *  nx + j] = frameA[i][j];
      frameBPtr[i * nx + j] = frameB[i][j];
    }
  }

  predictionFrame p;
  createPredictionFrame(&p, frameBPtr, nx, ny, blkDim);
  printf("\n%s\n", predictionFrameStr(p));
  float sum = 0.0;
  for(int i = 0; i < p.num_blks; i++) {
    float val = findBestBlkMse(p, frameAPtr, &p.blks[i], extraSpan);
    int* mv = p.blks[i].motion_vector;
    sum += val;

    #ifdef DEBUG
    #if (DEBUG > 0)
    printf("Blk %s, score: %.3f, mv: [%d, %d]\n", blkStr(p.blks[i]), val, mv[0], mv[1]);
    #endif
    #endif
  }
  printf("Score: %.3f\n", sum);

  int* motionCompFrame = motionCompensatedFrame(p);
  printf("\nFrame motionComp supposed to look like frameA:\n");
  printArrFrame(motionCompFrame, nx, ny);
}
