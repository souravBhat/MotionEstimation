#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda.h>
#include <math.h>
#include "device_launch_parameters.h"

// time stamp function in seconds
double getTimeStamp()
{
	struct timeval tv ;
	gettimeofday( &tv, NULL ) ;
	return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}

float getLuminence(int m, int n, float *block)
{
	float sum;
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<m; j++)
		{
			sum += block[i*m +j];
 		}
	}
	return sum;
}

float getContrast(int m, int n, float *block, float mu)
{
	float sum;
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<m; j++)
		{
			sum += (block[i*m +j] - mu)*(block[i*m +j] - mu);
 		}
	}
	return sum;
}

float getSigma_AB(int m, int n, float *blockA, float muA, float *blockB, float muB)
{
	float sum;
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<m; j++)
		{
			sum += (blockA[i*m +j] - muA)*(blockB[i*m +j] - muB);
 		}
	}
	return sum;
}

float get_SSIM(float *blockA, float *blockB, int m, int n, int N)
{
  int C1, C2, C3;
	float luminence, contrast, sigma_AB, structure, SSIM;

	float muA, luminenceA, muB, luminenceB;
	luminenceA = getLuminence(m,n,blockA);
	muA = luminenceA/N;
	luminenceB = getLuminence(m,n,blockB);
	muB = luminenceB/N;
	//printf("muA     %f     muB     %f \n",muA,muB);

	float sigmaA, contrastA, sigmaB, contrastB;
	contrastA = getContrast(m,n,blockA,muA);
	sigmaA = sqrt(contrastA/(N-1));
	contrastB = getContrast(m,n,blockB,muB);
	sigmaB = sqrt(contrastB/(N-1));
	//printf("sigmaA     %f     sigmaB     %f \n",sigmaA,sigmaB);

  C1=2; C2=2; C3=1;
	luminence = (2*muA*muB + C1)/(muA*muA + muB*muB + C1);
	contrast = (2*sigmaA*sigmaB + C2)/(sigmaA*sigmaA + sigmaB*sigmaB +C2);

	sigma_AB = getSigma_AB(m,n,blockA,muA,blockB,muB);
	sigma_AB = sigma_AB/(N-1);
	//printf("sigma_AB     %f\n",sigma_AB);

	structure = (sigma_AB + C3)/(sigmaA*sigmaB +C3);
	//printf("luminence     %f     contrast     %f     structure     %f \n",luminence,contrast,structure);

	SSIM = luminence * contrast * structure;
	return SSIM;

}

void initBlock(int m,int n, float *block)
{
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<m; j++)
		{
			block[i*m +j] = (rand()%10)+10;
 		}
	}
}

int main(int argc, char *argv[])
{
	int m=16;
	int n=16;
	int N=m*n;
	int bytes = N  * sizeof(float) ;
	float *blockA = (float *) malloc( bytes ) ;
	float *blockB = (float *) malloc( bytes ) ;
	float ssim_value;
	initBlock(m,n,blockA);
	initBlock(m,n,blockB);
	ssim_value=get_SSIM(blockA,blockB,m,n,N);
	printf("SSIM VALUE OBTAINED IS %f \n",ssim_value);
}
