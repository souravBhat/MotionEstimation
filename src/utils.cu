#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdint.h>
#include <inttypes.h>
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

int yuvReadFrame( FILE * const file,  uint8_t * const target_buffer ){

    if( fread(target_buffer,(3840*2160),1,file) != 1 ){
        printf("yuvReadFrame: The read was failed!\n");
        return 0;
    }
    return 1;
}

int yuvWriteToFile( char* file_name, int row, int col, uint8_t * const data_buffer) {

    FILE *file;

    file = fopen(file_name,"wb");
    if( file == NULL ){
        printf("yuvWriteToFile: Could not open the file %s\n", file_name);
        return 0;
    }
    printf("writing to file %s.\n",file_name);
    fwrite(data_buffer, 1, sizeof(row*col), file);
    fclose(file);
    return 1;
}