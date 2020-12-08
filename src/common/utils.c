#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include <sys/time.h>
#include "utils.h"
#include <math.h>

void printArrFrame(int * frame, int nx, int ny) {
  for(int y = 0; y < ny; y++) {
      for(int x = 0; x < nx; x++) {
        printf("%d\t", frame[y * nx + x]);
      }
      printf("\n");
  }
}

// Time stamp function in seconds.
double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}

FILE* yuvOpenInputFile( char* file_name) {

    FILE *file;


    file = fopen(file_name,"rb");
    if( file == NULL ){
        printf("yuvOpenInputFile: Could not open the file %s\n", file_name);
        return NULL;
    }

    /* Check the file size */
    fseek(file,0,SEEK_END);


    /* Set the file position back to the beginning of the file */
    fseek(file,0,SEEK_SET);
    return file;
}

void copyToIntBuffer(int* dest, uint8_t* source, int numElems) {
  for(int i = 0; i < numElems; i++){
    dest[i] = (int) source[i];
  }
}

void copyToUIntBuffer(uint8_t* dest, int* source, int numElems) {
  for(int i = 0; i < numElems; i++){
    dest[i] = (uint8_t) source[i];
  }
}

int yuvReadFrame(char *file_name,  int * const target_buffer, int numElems){
    FILE *file = yuvOpenInputFile(file_name);
    uint8_t *target_buffer_uint8 = (uint8_t*) malloc(sizeof(uint8_t) * numElems);
    if( fread(target_buffer_uint8,numElems,1,file) != 1 ){
        printf("yuvReadFrame: The read was failed!\n");
        return 0;
    }

    // Convert from uint8 to int array.
    copyToIntBuffer(target_buffer, target_buffer_uint8, numElems);
    free(target_buffer_uint8);
    return 1;
}

int yuvWriteFrame( char* file_name, int * const data_buffer, int numElems) {
    FILE *file;
    uint8_t* data_buffer_uint8 = (uint8_t*) malloc(sizeof(uint8_t) * numElems);
    copyToUIntBuffer(data_buffer_uint8, data_buffer, numElems);
    file = fopen(file_name,"wb");
    if( file == NULL ){
        int errnum = errno;
        printf("yuvWriteToFile: Could not open the file %s (%s)\n", file_name, strerror(errnum));
        return 0;
    }
    #ifdef DEBUG
    printf("writing to file %s.\n",file_name);
    #endif
    fwrite(data_buffer_uint8, 1, numElems, file);
    free(data_buffer_uint8);
    fclose(file);
    return 1;
}

void frameDiff(int* diffFrame, int* frameA, int* frameB, int numElems) {
  for(int i = 0; i < numElems; i++) {
    int diff = frameA[i] - frameB[i];
    if (diff < 0) diff = -diff;
    diffFrame[i] = diff;
  }
}

void motionCompensatedFrame(int* motionCompFrame, predictionFrame pf, int* ref_frame) {
  for(int i = 0; i < pf.num_blks; i++) {
      block blk = pf.blks[i];
      if(blk.is_best_match_found != 1) {
        printf("Error: Trying to create compensation frame without best match, value = %d for block %d\n", blk.is_best_match_found, i);
        exit(0);
      }

      int currFrameTopLeftX = blk.top_left_x;
      int currFrameTopLeftY = blk.top_left_y;
      int compFrameTopLeftX = currFrameTopLeftX + blk.motion_vectorX;
      int compFrameTopLeftY = currFrameTopLeftY + blk.motion_vectorY;
      // Populate the compensated frame.
      for(int offsetX = 0;  offsetX < blk.width; offsetX++) {
        for(int offsetY = 0; offsetY < blk.height; offsetY++) {
          // Get compensated position.
          int currX = currFrameTopLeftX + offsetX;
          int currY = currFrameTopLeftY + offsetY;
          int compX = compFrameTopLeftX + offsetX;
          int compY = compFrameTopLeftY + offsetY;
          // Populate array if within bounds.
          if(compX >= 0 && compY >= 0 && compX < pf.width && compY < pf.height) {
            #ifdef DEBUG
            #if (DEBUG > 1)
            printf("(%d, %d) moved to (%d, %d)\n", currX, currY, compX, compY);
            #endif
            #endif
            motionCompFrame[currY * pf.width + currX] = ref_frame[compY * pf.width + compX];
          }
        }
      }
    }
}

//Calculates image PSNR value
double imagePSNR(int *frame1, int *frame2, int x, int y)
{
    double MSE=0.0;
    double MSEtemp=0.0;
    double psnr=0.0;
    int index;
    int MAX = 0;
    //Calculate MSE
    for(index=0;index<x*y;index++)
    {
        if (MAX < frame1[index]){
            MAX = frame1[index];
        }
        if (MAX < frame2[index]){
            MAX = frame2[index];
        }
        MSEtemp=abs(frame1[index]-frame2[index]);
        MSE+=MSEtemp*MSEtemp;
    }
    MSE/=x*y;

    //Avoid division by zero
    if(MSE==0) return 99.0;

    psnr=20*log10(MAX) - 10*log10(MSE);

    return psnr;
}


