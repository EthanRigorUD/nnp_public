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
#include "kernals.h"




__global__ void matVecMulKer(const Matrix matrix, const float* vector, float* ouptut, float* bias, bool relu){
    //output vector should be size m*1 if matrix is m*n
    int outputLength = matrix.width;

}