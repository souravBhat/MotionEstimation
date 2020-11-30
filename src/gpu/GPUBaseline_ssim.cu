#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>

extern "C"{
    #include "../common/utils.h"
    #include "../common/block.h"
    #include "../common/prediction_frame.h"
}

// Given candidate block and current frame block, compute mse.
/*__device__ float computeMse(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk, int frameWidth){
    float sum = 0;
    for(int offsetY = 0; offsetY < blk.height; offsetY++) {
        for(int offsetX = 0; offsetX < blk.width; offsetX++) {
            int idxCand = (candBlkTopLeftY + offsetY) * frameWidth + (candBlkTopLeftX + offsetX);
            int idxRefBlk = (blk.top_left_y + offsetY) * frameWidth + (blk.top_left_x + offsetX);
            sum+= (predictionFrame[idxRefBlk] - referenceFrame[idxCand])*(predictionFrame[idxRefBlk] - referenceFrame[idxCand]);
        }
    }
    float score = sum / (blk.width * blk.height);
    return score;
}*/

__host__ __device__ float computeMean(int *Frame, int TopLeftX, int TopLeftY, block blk, int frameWidth)
{
  float sum = 0;
  for(int offsetY = 0; offsetY < blk.height; offsetY++) {
    for(int offsetX = 0; offsetX < blk.width; offsetX++) {
      int idx = (TopLeftY + offsetY) * frameWidth + (TopLeftX + offsetX);
      sum+= Frame[idx];
    }
  }
  sum = sum / (blk.width * blk.height);
  return sum;
}

__host__ __device__ float computeVar(int *Frame, int TopLeftX, int TopLeftY, block blk, int frameWidth, float Mean)
{
  float sum = 0;
  for(int offsetY = 0; offsetY < blk.height; offsetY++) {
    for(int offsetX = 0; offsetX < blk.width; offsetX++) {
      int idx = (TopLeftY + offsetY) * frameWidth + (TopLeftX + offsetX);
      sum+= (Frame[idx]-Mean)*(Frame[idx]-Mean);
    }
  }
  sum = sum / (blk.width * blk.height);
  return sum;
}

__host__ __device__ float computeCrossVar(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk, int frameWidth, int Meanref, int Meanpred)
{
  float sum = 0;
  for(int offsetY = 0; offsetY < blk.height; offsetY++) {
    for(int offsetX = 0; offsetX < blk.width; offsetX++) {
      int idxCand = (candBlkTopLeftY + offsetY) * frameWidth + (candBlkTopLeftX + offsetX);
      int idxRefBlk = (blk.top_left_y + offsetY) * frameWidth + (blk.top_left_x + offsetX);
      sum+= (referenceFrame[idxCand] - Meanref)*(predictionFrame[idxRefBlk] - Meanpred);
    }
  }
  sum = sum / (blk.width * blk.height);
  return sum;
}

// Given candidate block and current frame block, compute mse.
__device__ float computeSSIM(int *referenceFrame, int candBlkTopLeftX, int candBlkTopLeftY, int *predictionFrame, block blk, int frameWidth){
  float Meanref, Meanpred, Varref, Varpred, StdDevref, StdDevpred, CrossVar, Luminence, Contrast, Structure, score;
  //Increasing values of constants gives lower compensated score
  float C1 = 0.01; float C2 = 0.09; float C3 = 0.045;
  Meanref = computeMean(referenceFrame, candBlkTopLeftX, candBlkTopLeftY, blk, frameWidth);
  Meanpred = computeMean(predictionFrame, blk.top_left_x, blk.top_left_y, blk, frameWidth);
  Varref = computeVar(referenceFrame, candBlkTopLeftX, candBlkTopLeftY, blk, frameWidth, Meanref);
  Varpred = computeVar(predictionFrame, blk.top_left_x, blk.top_left_y, blk, frameWidth, Meanpred);
  StdDevref = sqrt(Varref);
  StdDevpred = sqrt(Varpred);
  CrossVar = computeCrossVar(referenceFrame, candBlkTopLeftX, candBlkTopLeftY, predictionFrame, blk, frameWidth, Meanref, Meanpred);
  Luminence = (2*Meanref*Meanpred + C1)/(Meanref*Meanref + Meanpred*Meanpred + C1);
  Contrast = (2*StdDevref*StdDevpred + C2)/(StdDevref*StdDevref + StdDevpred*StdDevpred + C2);
  Structure = (CrossVar + C3)/(StdDevref*StdDevpred + C3);
  score = Luminence*Contrast*Structure;
  return score;
}

// host side result

//device-side matrix addition

__global__ void f_findBestMatchBlock(int *currentframe, int *referenceframe,int extraSpan, block *block_list, int frameWidth, int frameHeight){
    //printf("%d\n",block_list[0].height);
    /* pick the block by using GPU block ID*/
    int blockID = blockIdx.y * ( frameWidth / block_list[0].width) + blockIdx.x;
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
    //printf("With ID = %d, candidateBlcokTopLeftX = %d, candidateBlcokTopLeftY = %d\n",threadID[ threadIdx.x + blockDim.x * threadIdx.y ],candidateBlcokTopLeftX,candidateBlcokTopLeftY);
    //printf("currentBlk.width = %d, frameWidth = %d, frameHeight = %d\n",currentBlk.width, frameWidth , frameHeight);


    result[threadIdx.x + threadIdx.y*blockDim.x] =0;
    if (candidateBlcokTopLeftX >= 0 && candidateBlcokBottomRightX <= frameWidth && candidateBlcokTopLeftY >= 0 && candidateBlcokBottomRightY <= frameHeight) {
        result[threadIdx.x + threadIdx.y * blockDim.x] = computeSSIM(referenceframe, candidateBlcokTopLeftX,
                                                                    candidateBlcokTopLeftY, currentframe, currentBlk, frameWidth);
        //printf("ID = %d, value = %lf\n", threadIdx.x + blockDim.x * threadIdx.y,
        //       result[threadIdx.x + threadIdx.y * blockDim.x]);
    }

    __syncthreads();

    /* calculating the minimum value in result array */
    unsigned int i = (nuBlocksWithinGPUGrid + 1) / 2;
    //printf("ID = %d, i = %d, nuBlocksWithinGPUGrid = %d\n", threadIdx.x + blockDim.x * threadIdx.y , i , nuBlocksWithinGPUGrid);

    while (i != 0) {
//            if (threadIdx.x + threadIdx.y * blockDim.x < i &&
//                threadIdx.x + threadIdx.y * blockDim.x + i < nuBlocksWithinGPUGrid) {
        if (threadIdx.x + threadIdx.y * blockDim.x < i &&
            threadIdx.x + threadIdx.y * blockDim.x + i < nuBlocksWithinGPUGrid) {
            //printf("comparing %lf to %lf \n",result[threadIdx.x + threadIdx.y*blockDim.x],result[threadIdx.x + threadIdx.y*blockDim.x + i]);
            if (result[threadIdx.x + threadIdx.y * blockDim.x] <
                result[threadIdx.x + threadIdx.y * blockDim.x + i]) {
                result[threadIdx.x + threadIdx.y * blockDim.x] = result[threadIdx.x + threadIdx.y * blockDim.x + i];
                threadID[threadIdx.x + threadIdx.y * blockDim.x] = threadID[threadIdx.x + threadIdx.y * blockDim.x +
                                                                            i];
            }
        }
        __syncthreads();
        i /= 2;
    }

    /* print out the best one result*/
#ifdef DEBUG
    if(threadIdx.x + threadIdx.y*blockDim.x == 0){
        //printf("smallest MSE = %lf with thread ID = %d\n",result[0], threadID[0]);
    }
#endif
    __syncthreads();

    /* compute the motion vector */
    if (threadIdx.x + threadIdx.y * blockDim.x == threadID[0]) {

        block_list[blockID].motion_vectorX = candidateBlcokTopLeftX - currentBlk.top_left_x;
        block_list[blockID].motion_vectorY = candidateBlcokTopLeftY - currentBlk.top_left_y;
        //printf("the  block has candidateBlcokTopLeftX = %d and currentBlk.top_left_x = %d and  motion vector x = %d\n",candidateBlcokTopLeftX, currentBlk.top_left_x, block_list[blockID].motion_vectorX);
#ifdef DEBUG
        printf("the %d block has motion vector x = %d, y = %d\n", blockID, block_list[blockID].motion_vectorX,
               block_list[blockID].motion_vectorY);
#endif

    }
    __syncthreads();

}



int main(int argc, char* argv[]) {

    if (argc < 4) {
        printf("Error: wrong number of argument. Usage: <current_frame> <reference_frame> <output_dir> [<blk_dim>] [<extra_span>] [<width>] [<height>]\n");
        exit(0);
    }
    char * currentFrameStr = argv[1];
    char * referenceFrameStr = argv[2];
    int blkDim = argc > 4 ? atoi(argv[4]) : 16;
    int extraSpan = argc > 5 ? atoi(argv[5]) : 15;
    int frameWidth =  argc > 6 ? atoi(argv[6]) : 3840;
    int frameHeight = argc > 7 ? atoi(argv[7]) : 2160;

    int numElems = frameWidth * frameHeight;
    int bytes = numElems * sizeof(int);

    // alloc memeory host-side
    int *h_referenceFrame, *h_currentFrame;

    /* allocate host side int pointer */
    cudaHostAlloc((void **) &h_referenceFrame, numElems * sizeof(int), 0);
    cudaHostAlloc((void **) &h_currentFrame, numElems * sizeof(int), 0);

    // Read current and reference frame.
    if(!yuvReadFrame(referenceFrameStr, h_referenceFrame, numElems)) {
        exit(1);
    };
    if(!yuvReadFrame(currentFrameStr, h_currentFrame, numElems)) {
        exit(1);
    }

    /* initialize host side block list pointer */
    block *h_block_list, *result_block_list;

    // Generate prediction frame, truncate into blocks.
    predictionFrame p;
    createPredictionFrame(&p, h_currentFrame, frameWidth, frameHeight, blkDim);

    /* allocate host side block list pointer and assign it to truncated block list*/
    cudaHostAlloc((void **) &h_block_list, p.num_blks * 48, 0);
    cudaHostAlloc((void **) &result_block_list, p.num_blks * 48, 0);
    h_block_list = p.blks;

#ifdef DEBUG
    printf("Number of blocks trauncated = %d\n", p.num_blks);
#endif

    // alloc memeory device-side
    int *d_referenceFrame, *d_currentDFrame;
    block *d_block_list;
    cudaMalloc(&d_referenceFrame, numElems * sizeof(int));
    cudaMalloc(&d_currentDFrame, numElems * sizeof(int));
    cudaMalloc(&d_block_list, p.num_blks * 48);

    double timeStampA = getTimeStamp() ;

    //transfer to device
    cudaMemcpy(d_currentDFrame, h_currentFrame, numElems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_referenceFrame, h_referenceFrame, numElems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_list, h_block_list, p.num_blks * 48, cudaMemcpyHostToDevice);

    double timeStampB = getTimeStamp() ;

    // invoke Kernel
    dim3 block(extraSpan*2 + 1, 2*extraSpan + 1);
#ifdef DEBUG
    printf("GPU block size (search window dimension) = %d \n",extraSpan*2 + 1);
    printf("frameWidth = %d and frameHeight = %d\n",frameWidth,frameHeight );
    printf("blkDim = %d \n",blkDim );
    printf("grid size x = %d and grid y = %d\n",(frameWidth + blkDim - 1) / blkDim,(frameHeight + blkDim - 1) / blkDim );
#endif
    dim3 grid((frameWidth + blkDim - 1) / blkDim, (frameHeight + blkDim - 1) / blkDim);
    //dim3 grid(1, 1);
    f_findBestMatchBlock<<<grid, block>>>(d_currentDFrame, d_referenceFrame, extraSpan,d_block_list, frameWidth, frameHeight);
    cudaDeviceSynchronize();

    double timeStampC = getTimeStamp() ;

    cudaMemcpy(result_block_list, d_block_list, p.num_blks * 48, cudaMemcpyDeviceToHost);

    double timeStampD = getTimeStamp() ;

    //printf("the first block has motion vector x = %d, y = %d\n",result_block_list[0].motion_vectorX,result_block_list[0].motion_vectorY);
    printf("%.6f %.6f %.6f %.6f\n",(timeStampD - timeStampA)*1000,(timeStampB - timeStampA)*1000, (timeStampC - timeStampB)*1000, (timeStampD - timeStampC)*1000 );
    p.blks = result_block_list;
#ifdef DEBUG
    for (int i = 0; i < 396; i++){
        printf("the %d block has motion vector x = %d, y = %d\n",i,result_block_list[i].motion_vectorX,result_block_list[i].motion_vectorY);

    }
#endif

    // Generate motion compensated frame and other results.

    int* outputFile = (int*) malloc(bytes * 5);
    memcpy(outputFile, h_referenceFrame, bytes);
    memcpy(&outputFile[numElems], h_currentFrame, bytes);
    motionCompensatedFrame(&outputFile[numElems*2], p, h_referenceFrame);
    // Difference between current and reference frames.
    frameDiff(&outputFile[numElems*3], h_referenceFrame, h_currentFrame, numElems);
    // Difference between current and motion compensated frames.
    frameDiff(&outputFile[numElems*4], &outputFile[numElems*2], h_currentFrame, numElems);

    // Compare MSE score with the motion compensated frame.
    float motionCompScore = 0.0;
    float originalScore = 0.0;
    for(int i =0; i < numElems; i++) {
        motionCompScore += (outputFile[numElems*2 + i] - h_currentFrame[i]) * (outputFile[numElems*2 + i]- h_currentFrame[i]);
        originalScore += (h_currentFrame[i] - h_referenceFrame[i]) * (h_currentFrame[i] - h_referenceFrame[i]);
    }

    // Output the frames of interest.
    #ifdef OUTPUT_FRAMES
    char * outputPath = argv[3];
    yuvWriteFrame(outputPath, outputFile, numElems*5);
    #endif

    cudaFree(d_referenceFrame);
    cudaFree(d_currentDFrame);
    cudaFree(d_block_list);
    cudaFreeHost(h_referenceFrame);
    cudaFreeHost(h_currentFrame);
    cudaFreeHost(result_block_list);
    cudaDeviceReset();

}
