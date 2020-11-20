#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>

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

__global__ void f_findBestMatchBlock(int *currentframe, int *referenceframe,int extraSpan, block *block_list, int nx, int ny){
    //printf("%d\n",block_list[0].height);
    /* pick the block by using GPU block ID*/
    int blockID = blockIdx.y * ( nx / block_list[0].width) + blockIdx.x;
    block currentBlk = block_list[blockID];

    /* computing the candidate block location by using thread ID*/
    int candidateBlcokTopLeftX = currentBlk.top_left_x - extraSpan + threadIdx.x;
    int candidateBlcokTopLeftY = currentBlk.top_left_y - extraSpan + threadIdx.y;
    int candidateBlcokBottomRightX = candidateBlcokTopLeftX + currentBlk.width - 1;
    int candidateBlcokBottomRightY = candidateBlcokTopLeftY + currentBlk.width - 1;
    int nuBlocksWithinGPUGrid = (extraSpan * 2 + 1) * (extraSpan * 2 + 1);

    /* shared memeory for saving the computed MSE result for each candidate block */
    __shared__ float result[ 1024 ];
    __shared__ int threadID[ 1024 ];

    threadID[ threadIdx.x + blockDim.x * threadIdx.y ] =threadIdx.x + blockDim.x * threadIdx.y;
    result[threadIdx.x + threadIdx.y*blockDim.x] =INT_MAX;
    if (candidateBlcokTopLeftX >= 0 && candidateBlcokBottomRightX <= nx && candidateBlcokTopLeftY >= 0 && candidateBlcokBottomRightY <= ny){
        result[threadIdx.x + threadIdx.y*blockDim.x] = computeMse(referenceframe, candidateBlcokTopLeftX, candidateBlcokTopLeftY, currentframe, currentBlk);
        //printf("ID = %d, value = %lf\n",threadID[ threadIdx.x + blockDim.x * threadIdx.y ], result[threadIdx.x + threadIdx.y*blockDim.x]);
    }

    __syncthreads();

    /* calculating the minimum value in result array */
    unsigned int i = (nuBlocksWithinGPUGrid + 1)/2;
    while(i != 0){
        if(threadIdx.x + threadIdx.y*blockDim.x < i && threadIdx.x + threadIdx.y*blockDim.x + i < nuBlocksWithinGPUGrid ){
            //printf("comparing %lf to %lf \n",result[threadIdx.x + threadIdx.y*blockDim.x],result[threadIdx.x + threadIdx.y*blockDim.x + i]);
            if (result[threadIdx.x + threadIdx.y*blockDim.x] > result[threadIdx.x + threadIdx.y*blockDim.x + i]){
                result[threadIdx.x + threadIdx.y*blockDim.x] = result[threadIdx.x + threadIdx.y*blockDim.x + i];
                threadID[threadIdx.x + threadIdx.y*blockDim.x] = threadID[threadIdx.x + threadIdx.y*blockDim.x + i];
            }
        }
        __syncthreads();
        i /= 2;
    }

    /* print out the best one result*/
#ifdef DEBUG
    if(threadIdx.x + threadIdx.y*blockDim.x == 0){
        printf("smallest MSE = %lf with thread ID = %d\n",result[0], threadID[0]);
    }
#endif
    __syncthreads();

    /* compute the motion vector */
    if(threadIdx.x + threadIdx.y*blockDim.x == threadID[0]){

        block_list[blockID].motion_vectorX = candidateBlcokTopLeftX - currentBlk.top_left_x;
        block_list[blockID].motion_vectorY = candidateBlcokTopLeftY - currentBlk.top_left_y;
        //printf("the  block has motion vector x = %d, y = %d\n",block_list[blockID].motion_vectorX,block_list[blockID].motion_vectorY);

    }
    __syncthreads();
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
    int extraSpan = 15;
    printf("block dimension = %d\n", blkDim);
    printf("extraSpan  = %d\n", extraSpan);

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
    block *h_block_list, *result_block_list;

    /* allocate host side int pointer */
    cudaHostAlloc((void **) &h_referenceFrame, noElems * sizeof(int), 0);
    cudaHostAlloc((block **) &h_currentDFrame, noElems * sizeof(int), 0);

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
    cudaHostAlloc((void **) &result_block_list, p.num_blks * 48, 0);
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
//    printf("%p\n", &h_block_list[0]);
//    printf("%p\n", &h_block_list[1]);
    printf("Number of blocks trauncated = %d\n", p.num_blks);


    // alloc memeory device-side
    int *d_referenceFrame, *d_currentDFrame;
    block *d_block_list;
    cudaMalloc(&d_referenceFrame, noElems * sizeof(int));
    cudaMalloc(&d_currentDFrame, noElems * sizeof(int));
    cudaMalloc(&d_block_list, p.num_blks * 48);

    double timeStampA = getTimeStamp() ;

    //transfer to device
    cudaMemcpy(d_currentDFrame, h_currentDFrame, noElems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_referenceFrame, h_referenceFrame, noElems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_list, h_block_list, p.num_blks * 48, cudaMemcpyHostToDevice);

    double timeStampB = getTimeStamp() ;

    // invoke Kernel
    dim3 block(extraSpan*2 + 1, 2*extraSpan + 1);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    //dim3 grid(1, 1);
    f_findBestMatchBlock<<<grid, block>>>(d_currentDFrame, d_referenceFrame, extraSpan,d_block_list, nx, ny);
    cudaDeviceSynchronize();

    double timeStampC = getTimeStamp() ;

    cudaMemcpy(result_block_list, d_block_list, p.num_blks * 48, cudaMemcpyDeviceToHost);

    double timeStampD = getTimeStamp() ;

    printf("the first block has motion vector x = %d, y = %d\n",result_block_list[0].motion_vectorX,result_block_list[0].motion_vectorY);

    printf("%.6f %.6f %.6f %.6f\n",(timeStampD - timeStampA)*1000,(timeStampB - timeStampA)*1000, (timeStampC - timeStampB)*1000, (timeStampD - timeStampC)*1000 );
    //printf("totoal= %.6f ms CPU_GPU_transfer = %.6f ms kernel =%.6f ms GPU_CPU_transfer= %.6f ms\n",(timeStampD - timeStampA)*1000,(timeStampB - timeStampA)*1000, (timeStampC - timeStampB)*1000, (timeStampD - timeStampC)*1000  );

    cudaFree(d_referenceFrame);
    cudaFree(d_currentDFrame);
    cudaFree(d_block_list);
    cudaFreeHost(h_referenceFrame);
    cudaFreeHost(h_currentDFrame);
    cudaFreeHost(result_block_list);
    cudaDeviceReset();

}


