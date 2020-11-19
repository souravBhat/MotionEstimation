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

// Given candidate block and current frame block, compute mse.
__device__ float computeMse(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk){
    float sum = 0;
    for(int offsetY = 0; offsetY < blk.height; offsetY++) {
        for(int offsetX = 0; offsetX <= blk.width; offsetX++) {
            int idxCand = (candBlkTopLeftY + offsetY) * blk.width + (candBlkTopLeftX + offsetX);
            int idxRefBlk = (blk.top_left_y + offsetY) * blk.width + (blk.top_left_x + offsetX);
            sum+= (predictionFrame[idxRefBlk] - referenceFrame[idxCand])*(predictionFrame[idxRefBlk] - referenceFrame[idxCand]);
        }
    }
    return sum/(blk.width * blk.height);
}


// host side result

//device-side matrix addition

__global__ void f_findBestMatchBlock(int *currentframe, int *referenceframe,int extraSpan, block *block_list ){
    //printf("%d\n",block_list[0].height);
    /* pick the block by using GPU block ID*/
    int blockID = blockIdx.y * ( 3840 / block_list[0].width) + blockIdx.x;
    block currentBlk = block_list[blockID];

    /* computing the candidate block by using thread ID*/
    int windowTopLeftX = currentBlk.top_left_x - extraSpan + threadIdx.x;
    int windowTopLeftY = currentBlk.top_left_y - extraSpan + threadIdx.y;
    int windowBottomRightX = windowTopLeftX + currentBlk.width - 1;
    int windowBottomRightY = windowTopLeftY + currentBlk.width - 1;
    int nuBlocksWithinGPUGrid = (extraSpan * 2) * (extraSpan * 2);

    /* shared memeory for saving the computed MSE result for each candidate block */
    __shared__ float result[ 1024 ];
    result[threadIdx.x + threadIdx.y*blockDim.x] =9999;
    if (windowTopLeftX >= 0 && windowBottomRightX <= 3840 && windowTopLeftY >= 0 && windowBottomRightY <= 2160){
        result[threadIdx.x + threadIdx.y*blockDim.x] = computeMse(referenceframe, windowTopLeftX, windowTopLeftY, currentframe, currentBlk);
        //printf("computed %lf\n",result[threadIdx.x + threadIdx.y*blockDim.x]);
    }

    __syncthreads();

    /* calculating the minimum value in result array */
    unsigned int i = nuBlocksWithinGPUGrid/2;
    while(i != 0){
        if(threadIdx.x + threadIdx.y*blockDim.x < i){
            //printf("comparing %lf to %lf \n",result[threadIdx.x + threadIdx.y*blockDim.x],result[threadIdx.x + threadIdx.y*blockDim.x + i]);
            result[threadIdx.x + threadIdx.y*blockDim.x] = fminf(result[threadIdx.x + threadIdx.y*blockDim.x], result[threadIdx.x + threadIdx.y*blockDim.x + i]);
        }
        __syncthreads();
        i /= 2;
    }

    /* print out the best one result*/
    if(threadIdx.x + threadIdx.y*blockDim.x == 0){
        printf("%lf\n",result[0]);
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




int main(int argc, char* argv[]) {

    if (argc != 3) {
        printf("Error: wrong number of argument\n");
        exit(0);
    }
    int nx = 3840;
    int ny = 2160;
    int blkDim = 16;
    int noElems = nx * ny;
//    int bytes_uint8 = noElems * sizeof(uint8_t);
//    int bytes_int = noElems * sizeof(int);
    int extraSpan = 2;

    // alloc memeory host-side
    uint8_t *h_uint8_referenceFrame = (uint8_t *) malloc(noElems * sizeof(uint8_t));
    uint8_t *h_uint8_currentDFrame = (uint8_t *) malloc(noElems * sizeof(uint8_t));
    printf("total bytes = %d\n", noElems * sizeof(uint8_t));

    printf("reference frame file name = %s\n", argv[1]);
    FILE *f = yuvOpenInputFile(argv[1]);
    yuvReadFrame(f,h_uint8_referenceFrame);
    fclose(f);

    printf("current frame file name = %s\n", argv[2]);
    f = yuvOpenInputFile(argv[2]);
    yuvReadFrame(f,h_uint8_currentDFrame);
    fclose(f);

    /* initialize host side int pointer */
    int *h_referenceFrame;
    int *h_currentDFrame;
    block *h_block_list;

    /* allocate host side int pointer */
    cudaHostAlloc((void **) &h_referenceFrame, noElems * sizeof(int), 0);
    cudaHostAlloc((void **) &h_currentDFrame, noElems * sizeof(int), 0);

    /* copy data from uint8 pointer to int pointer */
    for (int i = 0; i < noElems; i ++){
        *(h_currentDFrame + i) = (int) *(h_uint8_currentDFrame + i);
        *(h_referenceFrame + i) = (int) *(h_uint8_referenceFrame + i);
    }

    /* truncate the current frame to block list */
    predictionFrame p;
    createPredictionFrame(&p, h_currentDFrame, nx, ny, blkDim);

    /* allocate host side block list pointer and assign it to truncated block list*/
    cudaHostAlloc((void **) &h_block_list, p.num_blks * 48, 0);
    h_block_list = p.blks;

//    char *f_output = "./output/Jockey_3840x2160YF2.yuv";
//    yuvWriteToFile( f_output, ny, nx, h_currentDFrame);
//    uint8_t *ptr = h_currentDFrame;
//    for (int i = 0; i < 100; i ++ ){
//        printf("%" PRIu8 "\n", *(ptr++));
//    }
    //pin memory in host side
    //truncate frame to blocks

    //visualize the size of one block object
    printf("%p\n", &h_block_list[0]);
    printf("%p\n", &h_block_list[1]);
    printf("%d\n", p.num_blks);

    // alloc memeory device-side
    int *d_referenceFrame, *d_currentDFrame;
    block *d_block_list;
    cudaMalloc(&d_referenceFrame, noElems * sizeof(int));
    cudaMalloc(&d_currentDFrame, noElems * sizeof(int));
    cudaMalloc(&d_block_list, p.num_blks * 48);

    //transfer to device
    cudaMemcpy(d_currentDFrame, h_currentDFrame, noElems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_referenceFrame, h_referenceFrame, noElems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_list, h_block_list, p.num_blks * 48, cudaMemcpyHostToDevice);

    // invoke Kernel
    dim3 block(extraSpan*2, 2*extraSpan); // you will want to configure this
    //dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    dim3 grid(1, 1);
    f_findBestMatchBlock<<<grid, block>>>(d_currentDFrame, d_referenceFrame, extraSpan,d_block_list);
    cudaDeviceSynchronize();



    cudaFree(d_referenceFrame);
    cudaFree(d_currentDFrame);
    cudaFreeHost(h_referenceFrame);
    cudaFreeHost(h_currentDFrame);
    cudaDeviceReset();

}


