#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <inttypes.h>
#include "YUVreadfile.h"
#include "YUVwritefile.h"
#include "./../src/block.h"
#include "./../src/prediction_frame.h"

// time stamp function in seconds
double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}

// host side result

//device-side matrix addition
__global__ void f_findBestMatchBlock( uint8_t *referenceframe, block *blkList,int extraSpan ){
    currentBlk = blkList[blockIdx] //for every warp, we are going to take one current block from block list
    int topLeftX = currentBlk->top_left_x;
    int topLeftY = currentBlk->top_left_y;
    int windowTopLeftX = (topLeftX - extraSpan) < 0 ? 0 : topLeftX - extraSpan;
    int windowTopLeftY = (topLeftY - extraSpan) < 0 ? 0 : topLeftY - extraSpan;
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*blocksize + ix;

    int candBlkTopLeftX = threadIdx.x;
    int candBlkTopLeftY = threadIdx.y;

    float candMse = computeMse(referenceFrame, candBlkTopLeftX, candBlkTopLeftY, pf.frame, blk);

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




int main(int argc, char* argv[]) {

    if (argc != 3) {
        printf("Error: wrong number of argument\n");
        exit(0);
    }
    int nx = 3840;
    int ny = 2160;
    int blkDim = 16;
    int noElems = nx * ny;
    int bytes = noElems * sizeof(uint8_t);
    int extraSpan = 2;

    // alloc memeory host-side
    uint8_t *h_referenceFrame = (uint8_t *) malloc(bytes);
    uint8_t *h_currentDFrame = (uint8_t *) malloc(bytes);
    printf("total bytes = %d\n", bytes);

    printf("file name = %s\n", argv[1]);
    FILE *f = yuvOpenInputFile(argv[1]);
    yuvReadFrame(f,h_currentDFrame);
    fclose(f);

    predictionFrame p;
    createPredictionFrame(&p, h_currentDFrame, nx, ny, blkDim);

    printf("file name = %s\n", argv[2]);
    f = yuvOpenInputFile(argv[2]);
    yuvReadFrame(f,h_currentDFrame);
    fclose(f);

    char *f_output = "./output/Jockey_3840x2160YF2.yuv";
    yuvWriteToFile( f_output, ny, nx, h_currentDFrame);
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
    dim3 block(extraSpan, extraSpan); // you will want to configure this
    //dim3 grid((3840 + block.x - 1) / block.x, (2160 + block.y - 1) / block.y);
    dim3 grid(1, 1);
    //f_findBestMatchBlock<<<grid, block>>>(d_referenceFrame, d_currentDFrame, n);
    cudaDeviceSynchronize();



    cudaFree(d_referenceFrame);
    cudaFree(d_currentDFrame);
    cudaFreeHost(h_referenceFrame);
    cudaFreeHost(h_currentDFrame);
    cudaDeviceReset();

}


