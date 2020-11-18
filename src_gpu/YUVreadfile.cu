#include "YUVreadfile.h"


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