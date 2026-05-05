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

__global__ void matVecMulKer(const Matrix matrix, const float* vector, float* ouptut, bool relu){
    
}