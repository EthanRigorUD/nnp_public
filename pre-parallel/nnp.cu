/*
    nnp.cu

    Created on: Nov 9, 2025
    Serial implementation of a simple feedforward neural network for MNIST digit classification.

    Network architecture:
    - Input layer: 784 neurons (28x28 pixels)
    - Hidden layer 1: 128 neurons, ReLU activation
    - Hidden layer 2: 64 neurons, ReLU activation
    - Output layer: 10 neurons, Softmax activation

    Training:
    - Loss function: Categorical Cross-Entropy
    - Optimizer: Stochastic Gradient Descent (SGD)
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"
#include "kernels.h"


// old relu location

/* Derivative of ReLU activation function
* Arguments:
*   y: output value from ReLU function
* Returns:
*   derivative value
*/
float drelu(float y) { return y > 0 ? 1 : 0; }

/* Softmax activation function
* Arguments:
*   z: input array
*   out: output array to store softmax results
*   len: length of the input/output arrays
*/ 
void softmax(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

/* Initialize weights with small random values
* Arguments:
*   w: weight array to initialize
*   size: number of weights
*/
void init_weights(float *w, int size) {
    for (int i=0;i<size;i++)
        w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
}

//get the amount of blocks needed for kernal invocation
// ceiling method
float getBlocks(int x, int threads){
    int blocks = ceil((x + threads) / threads);
    return blocks;
}


/* Train the model using stochastic gradient descent 
* Arguments:
*   model (out): pointer to the MODEL structure which holds network parameters. It is populated by this function.
* Returns:
*   None
*/
void train_model(MODEL* model){
    init_weights(model->W1, SIZE*H1); init_weights(model->b1, H1);
    init_weights(model->W2, H1*H2); init_weights(model->b2, H2);
    init_weights(model->W3, H2*CLASSES); init_weights(model->b3, CLASSES);

    //weights and biases

    size_t size;
    Matrix w1 = {SIZE, H1, model->W1}; Matrix D_w1; float* D_b1; float* D_h1; float* D_h1a;
    D_w1.width = w1.width; D_w1.height = w1.height;
    size = SIZE*H1 * sizeof(float); cudaMalloc(&D_w1.elements, size);
    cudaMemcpy(D_w1.elements, w1.elements, size, cudaMemcpyHostToDevice);
    size = H1 * sizeof(float); cudaMalloc(&D_b1, size);
    
    Matrix w2 = {H1, H2, model->W2}; Matrix D_w2; float* D_b2;
    D_w2.width = w2.width; D_w2.height = w2.height;
    size = H1*H2 * sizeof(float); cudaMalloc(&D_w2.elements, size);
    cudaMemcpy(D_w2.elements, w2.elements, size, cudaMemcpyHostToDevice);
    size = H2 * sizeof(float); cudaMalloc(&D_b2, size);
    
    Matrix w3 = {H2, CLASSES, model->W3}; Matrix D_w3; float* D_b3;
    D_w3.width = w3.width; D_w3.height = w3.height;
    size = H2*CLASSES * sizeof(float); cudaMalloc(&D_w3.elements, size);
    cudaMemcpy(D_w3.elements, w3.elements, size, cudaMemcpyHostToDevice);
    size = CLASSES * sizeof(float); cudaMalloc(&D_b3, size);

    //other stuff
    float* D_train_data_row;
    float* D_output;
    size = SIZE * sizeof(float); cudaMalloc(&D_train_data_row, size);
    size = CLASSES * sizeof(float); cudaMalloc(&D_output, size);

    int threads = 256;
    for (int epoch=0; epoch<EPOCHS; epoch++) {
        float loss=0;
        for (int n=0; n<NUM_TRAIN; n++) {
            // ---------- Forward ----------
            float h2[H2], h2a[H2], h1[H1], h1a[H1];
            // old serial
            // float h1[H1], h1a[H1];
            // for (int j=0;j<H1;j++){
            //     h1[j]=model->b1[j];
            //     for (int i=0;i<SIZE;i++) h1[j]+=train_data[n][i]*model->W1[i*H1+j];
            //     h1a[j]=relu(h1[j]);
            // }
            // float h2[H2], h2a[H2];
            // for (int j=0;j<H2;j++){
            //     h2[j]=model->b2[j];
            //     for (int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j];
            //     h2a[j]=relu(h2[j]);
            // }
            // float out[CLASSES], outa[CLASSES];
            // for (int k=0;k<CLASSES;k++){
            //     out[k]=model->b3[k];
            //     for (int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k];
            // }
            // softmax(out,outa,CLASSES);

            size = sizeof(float) * SIZE;
            float out[CLASSES], outa[CLASSES];
            cudaMemcpy(D_train_data_row, train_data[n], size, cudaMemcpyHostToDevice);
            
            matVecMulKer<<<getBlocks(H1, threads), threads>>>(D_w1, D_train_data_row, D_h1a, D_b1, false);
            matVecMulKer<<<getBlocks(H2, threads), threads>>>(D_w2, D_h1a, D_h2a, D_b2, false);
            matVecMulKer<<<getBlocks(CLASSES, threads), threads>>>(D_w3, D_h2a, D_output, D_b3, true);

            size = sizeof(float) * CLASSES;
            cudaMemcpy(out, D_output, size, cudaMemcpyDeviceToHost);
            softmax(out,outa,CLASSES);
            // ---------- Loss ----------
            for (int k=0;k<CLASSES;k++)
                loss -= train_label[n][k]*logf(outa[k]+1e-8f);

            // ---------- Backprop ----------
            float delta3[CLASSES];
            for (int k=0;k<CLASSES;k++)
                delta3[k] = train_label[n][k]-outa[k];

            float delta2[H2];
            for (int j=0;j<H2;j++){
                float err=0;
                for (int k=0;k<CLASSES;k++) err+=delta3[k]*model->W3[j*CLASSES+k];
                delta2[j]=err*drelu(h2a[j]);
            }

            float delta1[H1];
            for (int j=0;j<H1;j++){
                float err=0;
                for (int k=0;k<H2;k++) err+=delta2[k]*model->W2[j*H2+k];
                delta1[j]=err*drelu(h1a[j]);
            }

            // ---------- Update ----------
            for (int j=0;j<H2;j++)
                for (int k=0;k<CLASSES;k++)
                    model->W3[j*CLASSES+k]+=LR*delta3[k]*h2a[j];
            for (int k=0;k<CLASSES;k++) model->b3[k]+=LR*delta3[k];

            for (int j=0;j<H1;j++)
                for (int k=0;k<H2;k++)
                    model->W2[j*H2+k]+=LR*delta2[k]*h1a[j];
            for (int k=0;k<H2;k++) model->b2[k]+=LR*delta2[k];

            for (int i=0;i<SIZE;i++)
                for (int j=0;j<H1;j++)
                    model->W1[i*H1+j]+=LR*delta1[j]*train_data[n][i];
            for (int j=0;j<H1;j++) model->b1[j]+=LR*delta1[j];
        }
        printf("Epoch %d, Loss=%.4f\n", epoch, loss/NUM_TRAIN);
    }
}

/* Save the trained model to a binary file
* Arguments:
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None
*/
void save_model(MODEL* model){
	FILE *f = fopen("model.bin", "wb");
	fwrite(model->W1, sizeof(float), SIZE*H1, f);
	fwrite(model->b1, sizeof(float), H1, f);
	fwrite(model->W2, sizeof(float), H1*H2, f);
	fwrite(model->b2, sizeof(float), H2, f);
	fwrite(model->W3, sizeof(float), H2*CLASSES, f);
	fwrite(model->b3, sizeof(float), CLASSES,f);
	fclose(f);
}

/* Load the trained model from a binary file
* Arguments:
*   model (out): pointer to the MODEL structure to populate with loaded weights and biases
* Returns:
*   None
*/
void load_model(MODEL* model){
	FILE *f = fopen("model.bin", "rb");
	fread(model->W1, sizeof(float), SIZE*H1, f);
	fread(model->b1, sizeof(float), H1, f);
	fread(model->W2, sizeof(float), H1*H2, f);
	fread(model->b2, sizeof(float), H2, f);
	fread(model->W3, sizeof(float), H2*CLASSES, f);
	fread(model->b3, sizeof(float), CLASSES, f);
	fclose(f);
}

/* Predict the class of a given input image
* Arguments:
*   x: input image array (flattened 28x28 pixels)
*   model: pointer to the MODEL structure containing trained weights and biases
* Returns:
*   None (prints predicted class and confidence)
*/
void predict(float *x, MODEL* model){
    float h1[H1], h1a[H1], h2[H2], h2a[H2], out[CLASSES], outa[CLASSES];

    // forward pass
    for (int j=0;j<H1;j++){ h1[j]=model->b1[j]; for(int i=0;i<SIZE;i++) h1[j]+=x[i]*model->W1[i*H1+j]; h1a[j]=relu(h1[j]); }
    for (int j=0;j<H2;j++){ h2[j]=model->b2[j]; for(int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j]; h2a[j]=relu(h2[j]); }
    for (int k=0;k<CLASSES;k++){ out[k]=model->b3[k]; for(int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k]; }
    softmax(out,outa,CLASSES);

    // print predicted class
    int pred=0; float max=outa[0];
    for(int k=1;k<CLASSES;k++) if(outa[k]>max){ max=outa[k]; pred=k; }
    printf("Predicted digit: %d (confidence %.2f)\n", pred, max);
}


