/* kernels.cu
 *
 *  Created on: Nov 9, 2025
 *  
 *  Location for CUDA kernels  kernels should be defined here, and prototypes placed in kernels.h
 *
 *  Example:
 *     __global__ void test_kernel(){}
 */
#include "nnp.h"
#include "kernels.h"


/* Activation functions for relu layers
* Arguments:
*   x: input value
* Returns:
*   activated value based on ReLU function 
*/
float relu(float x) { return x > 0 ? x : 0; }

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
        output[j] = relu(pos);
    }  
    else{
        output[j] = pos;
    }

}