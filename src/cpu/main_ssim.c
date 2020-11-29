#include <stdio.h>
#include <math.h>
#include "../common/block.h"
#include "../common/prediction_frame.h"
#include "../common/utils.h"
#include "../common/ssim.h"

void populateBlkMotionVector(block *blk, int dx, int dy) {
  blk->motion_vectorX = dx;
  blk->motion_vectorY = dy;
  blk->is_best_match_found = 1;
}

// Returns the score of the best block, given the window size.
float findBestBlkSSIM(predictionFrame pf, int *referenceFrame, block *blk, int extraSpan) {
  // Get bounds of search window.
  int topLeftX = blk->top_left_x;
  int topLeftY = blk->top_left_y;
  int bottomRightX = blk->bottom_right_x;
  int bottomRightY = blk->bottom_right_y;
  int windowTopLeftX = (topLeftX - extraSpan) < 0 ? 0 : topLeftX - extraSpan;
  int windowTopLeftY = (topLeftY - extraSpan) < 0 ? 0 : topLeftY - extraSpan;
  int windowBottomRightX = (bottomRightX + extraSpan) >= pf.width ? pf.width - 1 : (bottomRightX + extraSpan);
  int windowBottomRightY = (bottomRightY + extraSpan) >= pf.height ? pf.height - 1: (bottomRightY + extraSpan);

  float* match = findBestMatchSSIM(pf, referenceFrame, *blk, windowTopLeftX, windowTopLeftY, windowBottomRightX, windowBottomRightY);
  populateBlkMotionVector(blk, (int) match[1], (int)match[2]);

  return match[0];
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
  for(int i = 0; i < p.num_blks; i++) {
    float val = findBestBlkSSIM(p, refFrame, &p.blks[i], extraSpan);

    #ifdef DEBUG
    #if (DEBUG > 0)
    if (1) {
      printf("Blk %s, score: %.3f, mv: [%d, %d]\n", blkStr(p.blks[i]), val, p.blks[i].motion_vectorX, p.blks[i].motion_vectorY);
    }
    #endif
    #endif
  }

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
}
