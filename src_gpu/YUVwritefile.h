#ifndef YUVWRITEFILE_H
#define YUVWRITEFILE_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

int yuvWriteToFile( char* file_name, int row, int col, uint8_t * const data_buffer);
#endif