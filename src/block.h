#ifndef BLOCK_H
#define BLOCK_H
#include <stdio.h>
#include <stdlib.h>

typedef struct block {
  int idx_x;
  int idx_y;

  int top_left_x;
  int top_left_y;
  int bottom_right_x;
  int bottom_right_y;
  int width;
  int height;
  int is_best_match_found;
  int *motion_vector;
} block;

void createBlk(block *blk, int idxX, int idxY, int topleftX, int topLeftY, int bottomRightX, int bottomRightY);
char* blkStr(block blk);
#endif