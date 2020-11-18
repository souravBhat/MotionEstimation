#ifndef UTILS_H
#define UTILS_H


int yuvWriteToFile( char* file_name, int row, int col, uint8_t * const data_buffer);
void printArrFrame(int * frame, int width, int height);
FILE* yuvOpenInputFile( char* file_name);
int yuvReadFrame( FILE * const file, uint8_t * const target_buffer );
#endif