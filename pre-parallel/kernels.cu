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

__global__ void delta3Ker(float* delta,float* train,float* outa,int len){
    int j = blockIdx.x * blockDim.x + blockDim.x;
    if(j<len) delta[j] =train[j] - outa[j];
    //as shrimple as that
}

__global__ void delta12Ker(Matrix matrix,float* delta1,float* delta2,float*hLayer){
    int j = blockIdx.x * blockDim.x + blockDim.x;
    float err=0;

    if(j>=matrix.width) return;

    for (int k=0;k<matrix.height;k++) err+=delta2[k]*matrix.elements[j*matrix.height+k];

    //float drelu(float y) { return y > 0 ? 1 : 0; }
    int drelu = hLayer[j] > 0 ? 1: 0;
    delta1[j]=err*(drelu);

}

__global__ void updateWeights(Matrix matrix, float* vector, float* delta){
    int j = blockIdx.x * blockDim.x + blockDim.x;

    for (int k=0;k<matrix.width;k++) matrix.elements[j*matrix.width+k]+=LR*delta[k]*vector[j];

}
