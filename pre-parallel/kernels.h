/* 
 * kernels.h
 *
 *  Created on: Nov 9, 2025
 *  
 *  Placeholder Header file for CUDA kernel functions
*/

// Kernel function prototypes
//__global__ void test_kernel();

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

__global__ void matVecMulKer(const Matrix, const float*, float*, bool);
