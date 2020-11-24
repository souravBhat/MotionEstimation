#include "YUVwritefile.h"

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

