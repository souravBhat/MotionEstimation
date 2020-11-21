#ifndef UTILS_H
#define UTILS_H
#include "prediction_frame.h"

int yuvWriteFrame( char* file_name, int * const data_buffer, int numElems);
int yuvReadFrame(char *file_name,  int * const target_buffer, int numElems);
void frameDiff(int* diffFrame, int* frameA, int* frameB, int numElems);
void printArrFrame(int * frame, int width, int height);
int* motionCompensatedFrame(predictionFrame pf, int* ref_frame);
#endif