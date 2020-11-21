#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <string.h>
#include "utils.h"


void printArrFrame(int * frame, int nx, int ny) {
  for(int y = 0; y < ny; y++) {
      for(int x = 0; x < nx; x++) {
        printf("%d\t", frame[y * nx + x]);
      }
      printf("\n");
  }
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
    printf("writing to file %s.\n",file_name);
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

int* motionCompensatedFrame(predictionFrame pf, int* ref_frame) {
  int* motionCompFrame = (int*) malloc(sizeof(int) * pf.width * pf.height);
  for(int i = 0; i < pf.num_blks; i++) {
      block blk = pf.blks[i];
      if(!blk.is_best_match_found) continue;

      // Populate the compensated frame.
      for(int x = blk.top_left_x; x <= blk.bottom_right_x; x++) {
        for(int y = blk.top_left_y; y <= blk.bottom_right_y; y++) {
          // Get compensated position.
          int comp_ref_x = x - blk.motion_vectorX;
          int comp_ref_y = y - blk.motion_vectorY;
          // Populate array if within bounds.
          if(comp_ref_x >= 0 && comp_ref_y >= 0 && comp_ref_x < pf.width && comp_ref_y < pf.height) {
            #ifdef DEBUG
            #if (DEBUG > 0)
            printf("(%d, %d) moved to (%d, %d)\n", x, y, comp_ref_x, comp_ref_y);
            #endif
            #endif
            motionCompFrame[y * pf.width + x] = ref_frame[comp_ref_y * pf.width + comp_ref_x];
          }
        }
      }
    }
    return motionCompFrame;
}
