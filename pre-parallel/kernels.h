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
    int height;
    int width;
    float* elements;
} Matrix;



__global__ void matVecMulKer(const Matrix, const float*, float*, float*, bool);

__global__ void delta1Ker(float*,float*,float*,int);
__global__ void delta23Ker(float*,float*,float*,Matrix);
