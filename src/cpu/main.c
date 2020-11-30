#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include "../common/block.h"
#include "../common/prediction_frame.h"
#include "../common/utils.h"

void populateBlkMotionVector(block *blk, int dx, int dy) {
  blk->motion_vectorX = dx;
  blk->motion_vectorY = dy;
  blk->is_best_match_found = 1;
}

// Given candidate block and current frame block, compute mse.
float computeMse(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk, int frameWidth){
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
}

// Find best block using mse metric within the search window.
float* findBestMatchMse(predictionFrame pf, int *referenceFrame, block blk, int windowTopLeftX, int windowTopLeftY,
  int windowBottomRightX, int windowBottomRightY) {
  // First is the score, 2nd and 3rd element is the vector.
  float* result = (float*) malloc(sizeof(float) * 3);
  result[0] = INFINITY;

  #ifdef DEBUG
  #if (DEBUG > 0)
  if (blk.idx_x == 17 && blk.idx_y == 2) {
    printf("Blk %s, Search window: (%d, %d) -> (%d, %d)\n\n\n", blkStr(blk), windowTopLeftX, windowTopLeftY, windowBottomRightX, windowBottomRightY);
  }
  #endif
  #endif

  for(int y = windowTopLeftY; y <= windowBottomRightY - blk.height + 1; y++) {
    for(int x = windowTopLeftX; x <= windowBottomRightX - blk.width + 1; x++) {
      float candMse = computeMse(referenceFrame, x, y, pf.frame, blk,  pf.width);
      if(candMse < result[0]) {
        result[0] = candMse;
        result[1] = x - blk.top_left_x;
        result[2] = y - blk.top_left_y;
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

// Run config for pthread.
typedef struct runConfig {
  predictionFrame* p;
  int* refFrame;
  block* blk;
  int extraSpan;
  int score;
} runConfig;

void createRunConfig(runConfig* config, predictionFrame* p, int* refFrame, block* blk, int extraSpan) {
  config->p = p;
  config->refFrame = refFrame;
  config->blk = blk;
  config->extraSpan = extraSpan;
}

// Method called by the pthreads.
void *runFindBestBlkMse(void* args) {
  runConfig* config = (runConfig*) args;
  int val = findBestBlkMse(*config->p, config->refFrame, config->blk, config->extraSpan);
  config->score = val;
  pthread_exit(0);
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
      printf("Error: wrong number of argument. Usage: <current_frame> <reference_frame> <output_dir> [<blk_dim>] [<extra_span>] [<width>] [<height>]\n");
      exit(0);
  }
  char * currentFrameStr = argv[1];
  char * referenceFrameStr = argv[2];
  char * outputDir = argv[3];
  int blkDim = argc > 4 ? atoi(argv[4]) : 16;
  int extraSpan = argc > 5 ? atoi(argv[5]) : 7;
  int frameWidth =  argc > 6 ? atoi(argv[6]) : 3840;
  int frameHeight = argc > 7 ? atoi(argv[7]) : 2160;
  printf("[\n  Current Frame: %s\n  Reference Frame: %s\n  Output Dir: %s\n  BlkDim: %d\n  ExtraSpan: %d\n  FrameWidth: %d\n  FrameHeight: %d\n]\n",
    currentFrameStr, referenceFrameStr, outputDir, blkDim, extraSpan, frameWidth, frameHeight);

  int numElems = frameWidth * frameHeight;
  int bytes = sizeof(int) * numElems;

  // File locations for the results.
  char outputFileName[100];
  sprintf(outputFileName, "%s/output_%d_%d.yuv", outputDir, blkDim, extraSpan);

  // Read current and reference frame.
  int * currentFrame = (int*) malloc(bytes);
  int * refFrame = (int*) malloc(bytes);
  if(!yuvReadFrame(currentFrameStr, currentFrame, numElems)) {
    exit(1);
  };
  if(!yuvReadFrame(referenceFrameStr, refFrame, numElems)) {
    exit(1);
  }

  // Generate prediction frame, truncate into blocks and find best match mse block.
  predictionFrame p;
  createPredictionFrame(&p, currentFrame, frameWidth, frameHeight, blkDim);
  pthread_t tids[p.num_blks];

  double timeStampA = getTimeStamp();
  for(int i = 0; i < p.num_blks; i++) {
    runConfig *config = (runConfig*)malloc(sizeof(runConfig));
    createRunConfig(config, &p, refFrame, &p.blks[i], extraSpan);
    pthread_create(&tids[i], NULL, runFindBestBlkMse, config);
  }

  // Wait for threads to complete.
  for(int i = 0; i < p.num_blks; i++) pthread_join(tids[i], NULL);
  double timeStampB = getTimeStamp();

  // Generate motion compensated frame and other results.
  int* outputFile = (int*) malloc(bytes * 5);
  memcpy(outputFile, refFrame, bytes);
  memcpy(&outputFile[numElems], currentFrame, bytes);
  motionCompensatedFrame(&outputFile[numElems*2], p, refFrame);
  // Difference between current and reference frames.
  frameDiff(&outputFile[numElems*3], refFrame, currentFrame, numElems);
  // Difference between current and motion compensated frames.
  frameDiff(&outputFile[numElems*4], &outputFile[numElems*2], currentFrame, numElems);

  // Compare MSE score with the motion compensated frame.
  float motionCompScore = 0.0;
  float originalScore = 0.0;
  for(int i =0; i < numElems; i++) {
    motionCompScore += (outputFile[numElems*2 + i] - currentFrame[i]) * (outputFile[numElems*2 + i]- currentFrame[i]);
    originalScore += (currentFrame[i] - refFrame[i]) * (currentFrame[i] - refFrame[i]);
  }
  printf("Original Score: %.4f, Compensated Score: %.4f\n", originalScore/numElems, motionCompScore/numElems);

  // Output the frames of interest.
  printf("Output file dimensions: (%d x %d)\n", frameWidth, 5*frameHeight);
  yuvWriteFrame(outputFileName, outputFile, numElems*5);
  printf("Computation time: %.lf ms\n", (timeStampB - timeStampA)*1000);
}
