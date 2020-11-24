#ifndef YUVREADFILE_H
#define YUVREADFILE_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

FILE* yuvOpenInputFile( char* file_name);
int yuvReadFrame( FILE * const file, uint8_t * const target_buffer );
#endif