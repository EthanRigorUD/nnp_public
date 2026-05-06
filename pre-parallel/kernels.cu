/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */
#include "config.h"
#include "nnp.h"
#include "kernels.h"


//forward
__global__ void matVecMulKer(const Matrix matrix, const float* vector, float* output, float* bias, bool isRelu){
    //output vector should be size m*1 if matrix is m*n
    int outputLength = matrix.width;
    // when multiplying a matrix and vector, the height needs to be the same, ergo
    int inputLength = matrix.height;

    int j = blockIdx.x * blockDim.x + blockDim.x;

    float pos = bias[j];
    // serial inner loop
    for (int i=0;i<inputLength;i++) pos+=vector[i]*matrix.elements[i*outputLength+j];

    if (isRelu){
        output[j] = pos > 0 ? pos : 0;
    }  
    else{
        output[j] = pos;
    }

}
//delta

__global__ void delta1Ker(float*,float*,float*,int){
    int j = blockIdx.x * blockDim.x + blockDim.x;

}

__global__ void delta23Ker(float*,float*,float*,Matrix){
    int j = blockIdx.x * blockDim.x + blockDim.x;

}

