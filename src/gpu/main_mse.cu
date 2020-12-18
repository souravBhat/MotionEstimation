#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>
#define FULL_MASK 0xffffffff

extern "C"{
    #include "../common/utils.h"
    #include "../common/block.h"
    #include "../common/prediction_frame.h"
}

// Given candidate block and current frame block, compute mse.
__device__ float computeMse(
  int* window,
  int* currentFrameBlk,
  block blk,
  int candBlkTopLeftX,
  int candBlkTopLeftY,
  int frameWidth,
  int windowWidth,
  int windowTopLeftX,
  int windowTopLeftY
){
    float sum = 0;
    for(int offsetY = 0; offsetY < blk.height; offsetY++) {
        for(int offsetX = 0; offsetX < blk.width; offsetX++) {
            int idxCand = (candBlkTopLeftY - windowTopLeftY + offsetY) * windowWidth + (candBlkTopLeftX - windowTopLeftX + offsetX);
            int idxRefBlk = (offsetY) * blk.width + (offsetX);
            int diff = currentFrameBlk[idxRefBlk] - window[idxCand];
            sum+= diff * diff;
        }
    }
    float score = sum / (blk.width * blk.height);
    return score;
}


__global__ void f_findBestMatchBlock(int *currentframe, int *referenceFrame,int extraSpan, block *block_list, int frameWidth, int frameHeight, int numWindowElems, int blkDim){
    /* pick the block by using GPU block ID*/
    int blockID = blockIdx.y * ( frameWidth / block_list[0].width) + blockIdx.x;
    block currentBlk = block_list[blockID];

    /* computing the candidate block location by using thread ID*/
    int tid = threadIdx.x;
    int tidX = threadIdx.x % 32;
    int tidY = threadIdx.x / 32;
    int candBlkTopLeftX = currentBlk.top_left_x - extraSpan + tidX;
    int candBlkTopLeftY = currentBlk.top_left_y - extraSpan + tidY;
    int candBlkBottomRightX = candBlkTopLeftX + currentBlk.width - 1;
    int candBlkBottomRightY = candBlkTopLeftY + currentBlk.height - 1;
    int nuBlocksWithinGPUGrid = (extraSpan * 2 + 1) * (extraSpan * 2 + 1);

    // NOTE: These are hypothetical window indices for now and can be negative.
    // TODO(Sourav): Simplify the indexing into the shared memory array.
    int hypoWindowTopLeftX = currentBlk.top_left_x - extraSpan;
    int hypoWindowTopLeftY = currentBlk.top_left_y - extraSpan;
    int hypoWindowBottomRightX = currentBlk.bottom_right_x + extraSpan;
    int hypoWindowWidth= hypoWindowBottomRightX - hypoWindowTopLeftX + 1;

    int currentFrameOffsetX = threadIdx.x % blkDim;
    int currentFrameOffsetY = threadIdx.x / blkDim;
    int predBlkX = currentBlk.top_left_x + currentFrameOffsetX;
    int predBlkY = currentBlk.top_left_y + currentFrameOffsetY;

    // Shared memory for saving the computed MSE result for each candidate block.
    __shared__ float result[ 1024 ];
    __shared__ int threadID[ 1024 ];
    // Contains the window relevant for this particular block.
    extern __shared__ int window[];
    int *currentFrameBlk = &window[numWindowElems];

    // Initialize shared memory values.
    threadID[threadIdx.x] = threadIdx.x;
    result[threadIdx.x] = INT_MAX;
    // Copy the window values from the reference frame to the shared mem location.
    if (candBlkTopLeftX >= 0 && candBlkTopLeftY >= 0 && candBlkTopLeftX < frameWidth && candBlkTopLeftY < frameHeight) {
      window[threadIdx.x] = referenceFrame[candBlkTopLeftY * frameWidth + candBlkTopLeftX];
    }
    __syncthreads();

    if (predBlkX <= currentBlk.bottom_right_x && predBlkY <= currentBlk.bottom_right_y) {
      currentFrameBlk[threadIdx.x] = currentframe[predBlkY * frameWidth + predBlkX];
    }
    __syncthreads();
    // TODO: Fix thread divergence.
    // Also make sure the indices are within actual window bounds (NOT hypotheticals).
    if (candBlkTopLeftX >= 0 && candBlkBottomRightX < frameWidth && candBlkTopLeftY >= 0 && candBlkBottomRightY < frameHeight &&
            candBlkBottomRightX <= currentBlk.bottom_right_x + extraSpan &&
            candBlkBottomRightY <= currentBlk.bottom_right_y + extraSpan) {

        result[threadIdx.x ] = computeMse(
          window, currentFrameBlk, currentBlk,
          candBlkTopLeftX, candBlkTopLeftY, frameWidth, hypoWindowWidth,
          hypoWindowTopLeftX, hypoWindowTopLeftY
        );
    }
    __syncthreads();

    // Calculating the minimum value in result array.
    unsigned int i = (nuBlocksWithinGPUGrid + 1) / 2;
    int outerLimit = nuBlocksWithinGPUGrid;
    while (i != outerLimit && outerLimit >= 32) {
        int thisElem = threadIdx.x;
        int stepElem = threadIdx.x + i;
        if (thisElem  < i && stepElem < outerLimit) {
            if (result[thisElem] > result[stepElem]) {
                result[thisElem] = result[stepElem];
                threadID[thisElem] = threadID[stepElem];
            }
        }
        __syncthreads();
        outerLimit = i;
        i = (i + 1)/2;
    }
    __syncthreads();

    // Use warp level primitives to reduce the last 32 elements of the array.
    unsigned mask = __ballot_sync(FULL_MASK, tid < 32);
    if (tid < 32) {
      float resultVal = result[tid];
      int threadVal = threadID[tid];
      for (int offset = 16; offset > 0; offset /= 2) {
        float compResultVal = __shfl_down_sync(mask, resultVal, offset);
        float compThreadVal = __shfl_down_sync(mask, threadVal, offset);
        if (resultVal > compResultVal) {
          resultVal = compResultVal;
          threadVal = compThreadVal;
        }
      }
      // Thread 0 will have the minimum value and the threadVal corresponding to it.
      // The motion vectors can be calculated using the threadVal.
      if(tid == 0) {
        block_list[blockID].motion_vectorX = candBlkTopLeftX - currentBlk.top_left_x + threadVal % 32;
        block_list[blockID].motion_vectorY = candBlkTopLeftY - currentBlk.top_left_y + threadVal / 32;
        block_list[blockID].is_best_match_found = 1;
      }
    }
}



int main(int argc, char* argv[]) {

    if (argc < 4) {
        printf("Error: wrong number of argument. Usage: <current_frame> <reference_frame> <output_dir> [<blk_dim>] [<extra_span>] [<width>] [<height>]\n");
        exit(0);
    }
    char * currentFrameStr = argv[1];
    char * referenceFrameStr = argv[2];
    int blkDim = argc > 4 ? atoi(argv[4]) : 8;
    int extraSpan = argc > 5 ? atoi(argv[5]) : 12;
    int frameWidth =  argc > 6 ? atoi(argv[6]) : 3840;
    int frameHeight = argc > 7 ? atoi(argv[7]) : 2160;

    int numElems = frameWidth * frameHeight;
    int numWindowElems = (2 * extraSpan + blkDim) * (2 * extraSpan + blkDim);
    int numcurrentFrameBlk = blkDim*blkDim;
    int sharedmem = numcurrentFrameBlk + numWindowElems ;
    int bytes = numElems * sizeof(int);

    // alloc memory host-side
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

    // Define grid of 32 x 32 threads.
    dim3 block(1024);
    dim3 grid((frameWidth + blkDim - 1) / blkDim, (frameHeight + blkDim - 1) / blkDim);

    #ifdef DEBUG
    printf("GPU block size (search window dimension) = %d \n",extraSpan*2 + 1);
    printf("frameWidth = %d and frameHeight = %d\n",frameWidth,frameHeight );
    printf("blkDim = %d \n",blkDim );
    printf("grid size x = %d and grid y = %d\n",(frameWidth + blkDim - 1) / blkDim,(frameHeight + blkDim - 1) / blkDim );
    #endif

    // Invoke kernel.
    f_findBestMatchBlock<<<grid, block, sharedmem * sizeof(int)>>>(d_currentDFrame, d_referenceFrame, extraSpan,d_block_list, frameWidth, frameHeight, numWindowElems, blkDim);
    cudaDeviceSynchronize();
    double timeStampC = getTimeStamp() ;

    cudaMemcpy(result_block_list, d_block_list, p.num_blks * 48, cudaMemcpyDeviceToHost);

    double timeStampD = getTimeStamp() ;


    p.blks = result_block_list;

    #ifdef DEBUG
    #if (DEBUG > 1)
    for (int i = 0; i < 396; i++){
        printf("the %d block has motion vector x = %d, y = %d\n",i,result_block_list[i].motion_vectorX,result_block_list[i].motion_vectorY);

    }
    #endif
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
    float psnr = 0.0;
    psnr = imagePSNR(&outputFile[numElems*2], h_currentFrame, frameWidth, frameHeight);
    printf("%.6f %.6f %.6f %.6f %.6f\n",(timeStampD - timeStampA)*1000,(timeStampB - timeStampA)*1000, (timeStampC - timeStampB)*1000, (timeStampD - timeStampC)*1000, psnr );
    #ifdef DEBUG
    #if (DEBUG > 0)
    // Compare MSE score with the motion compensated frame.
    float motionCompScore = 0.0;
    float originalScore = 0.0;
    for(int i =0; i < numElems; i++) {
        motionCompScore += (outputFile[numElems*2 + i] - h_currentFrame[i]) * (outputFile[numElems*2 + i]- h_currentFrame[i]);
        originalScore += (h_currentFrame[i] - h_referenceFrame[i]) * (h_currentFrame[i] - h_referenceFrame[i]);
    }
    printf("Original score: %.3f, Compensated score: %.3f\n", originalScore/numElems, motionCompScore/numElems);
    #endif
    #endif

    // Output the frames of interest.
    #ifdef OUTPUT_FRAMES
    char * outputPath = argv[3];
    yuvWriteFrame(outputPath, outputFile, numElems*5);
    #endif

    cudaError_t err = cudaGetLastError();
    if (err) printf("Error: %s\n", cudaGetErrorString(err));

    cudaFree(d_referenceFrame);
    cudaFree(d_currentDFrame);
    cudaFree(d_block_list);
    cudaFreeHost(h_referenceFrame);
    cudaFreeHost(h_currentFrame);
    cudaFreeHost(result_block_list);
    cudaDeviceReset();

}
