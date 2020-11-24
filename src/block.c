#include "block.h"

void createBlk(block *blk, int idxX, int idxY, int topLeftX, int topLeftY, int width, int height) {
  blk->idx_x = idxX;
  blk->idx_y = idxY;
  blk->top_left_x = topLeftX;
  blk->top_left_y = topLeftY;
  blk->width = width;
  blk->height = height;
  blk->bottom_right_x = topLeftX + width - 1;
  blk->bottom_right_y = topLeftY + height - 1;
  blk->motion_vectorY = -1000;
}

char* blkStr(block blk) {
  char *c = (char*) malloc(sizeof(char) * 30);
  sprintf(c, "Blk (%d, %d) | (%d, %d) -> (%d, %d)", blk.idx_x, blk.idx_y,
    blk.top_left_x, blk.top_left_y, blk.bottom_right_x, blk.bottom_right_y);
  return c;
}
