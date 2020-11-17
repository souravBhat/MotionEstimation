#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <inttypes.h>

// time stamp function in seconds
double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}

// host side result

// device-side matrix addition
__global__ void f_findBestMatchBlock( float *reference, float *current, int blocksize, int offset ){

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*n + ix ;

    int partialSum = 0;
    startingIndex = offset + ix + iy*3840;
    uint8_t *ptr = reference + startingIndex;
    for (int i = 0; i < blocksize; i++){
        for (int j = 0; j < blocksize; j++){
            partialSum += *(ptr);
            ptr++;
        }
        ptr += 3840;
    }





}

float SumDataA(float* A, int n){
    double r = 0;
    float *ia = A;
    for (int i =0; i<n; i++){
        for (int j =0; j<n ; j++){
            for (int k =0; k<n ; k ++){
                r += ia[i*(n)*(n) + j * (n) + k] *(((i + j  + k)%2)?1:-1);

                //printf("at ix + iy + ix = %d, ia = %lf,  r = %lf\n",ix + iy + iz,ia[ix + iy*n + iz*n*n], r);
            }
        }
    }

    return (float)r;
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

int yuvReadFrame( FILE * const file,
                    uint8_t * const target_buffer ){

    if( fread(target_buffer,(3840*2160),1,file) != 1 ){
        printf("yuvReadFrame: The read was failed!\n");
        return 0;
    }
    return 1;
}


int main(int argc, char* argv[]) {

    if (argc != 3) {
        printf("Error: wrong number of argument\n");
        exit(0);
    }

    int noElems = 3840 * 2160;
    int bytes = noElems * sizeof(uint8_t);

    // alloc memeory host-side
    uint8_t *h_referenceFrame = (uint8_t *) malloc(bytes);
    uint8_t *h_currentDFrame = (uint8_t *) malloc(bytes);
    printf("total bytes = %d\n", bytes);

    printf("file name = %s\n", argv[1]);
    FILE *f = yuvOpenInputFile(argv[1]);
    yuvReadFrame(f,h_currentDFrame);
    fclose(f);

    printf("file name = %s\n", argv[2]);
    f = yuvOpenInputFile(argv[2]);
    yuvReadFrame(f,h_currentDFrame);
    fclose(f);

//    uint8_t *ptr = h_currentDFrame;
//    for (int i = 0; i < 100; i ++ ){
//        printf("%" PRIu8 "\n", *(ptr++));
//    }
    //pin memory in host side
    cudaHostAlloc((void **) &h_referenceFrame, bytes, 0);
    cudaHostAlloc((void **) &h_currentDFrame, bytes, 0);

    //truncate frame to blocks

    // alloc memeory device-side
    float *d_referenceFrame, *d_currentDFrame;
    cudaMalloc(&d_referenceFrame, bytes);
    cudaMalloc(&d_currentDFrame, bytes);



    // getting host side result


    //transfer to device
    cudaMemcpy(d_referenceFrame, h_referenceFrame, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_referenceFrame, h_referenceFrame, bytes, cudaMemcpyHostToDevice);


    int searchRange = 2;
    // invoke Kernel
    dim3 block(searchRange*2, searchRange*2); // you will want to configure this
    //dim3 grid((3840 + block.x - 1) / block.x, (2160 + block.y - 1) / block.y);
    dim3 grid(1, 1);
    f_findBestMatchBlock<<<grid, block>>>(d_referenceFrame, d_currentDFrame, n);
    cudaDeviceSynchronize();



    cudaFree(d_referenceFrame);
    cudaFree(d_currentDFrame);
    cudaFreeHost(h_referenceFrame);
    cudaFreeHost(h_currentDFrame);
    cudaDeviceReset();

}


