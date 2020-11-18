#ifndef UTILS_H
#define UTILS_H

FILE* yuvOpenInputFile(char* file_name);
int yuvReadFrame(FILE * const file, int * const target_buffer, int frameWidth, int frameHeight);
int yuvWriteFrame(char *fileName, int * buffer);
void printArrFrame(int * frame, int width, int height);
#endif