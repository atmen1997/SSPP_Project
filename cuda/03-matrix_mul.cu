// 1. Constants/Headers
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define BLOCK_SIZE 8

// 2. Tiled Matrix Multiplication Using Shared Memory Caculation
__global__ void matrixMulSharedCUDA(float *A, float *B, float *C, int n, int k)
{
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over sub-blocks of the input
    for (int t = 0; t < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load tile from A (if in range)
        if (row < n && (t * BLOCK_SIZE + threadIdx.x) < k) {
            Asub[threadIdx.y][threadIdx.x] = A[row * k + (t * BLOCK_SIZE + threadIdx.x)];
        } else {
            Asub[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B (if in range)
        if (col < n && (t * BLOCK_SIZE + threadIdx.y) < k) {
            Bsub[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * n + col];
        } else {
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the loaded tiles
        for (int l = 0; l < BLOCK_SIZE; l++) {
            sum += Asub[threadIdx.y][l] * Bsub[l][threadIdx.x];
        }
        __syncthreads();
    }

    // Write result if in valid output range
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// 3. CPU Matrix Multiplication Calculation
void matrixMultiplyCPU(float *A, float *B, float *C, int n, int k)
{
    for (int i = 0; i < n; i++) {           // Loop over rows of A and C 
        for (int l = 0; l < k; l++) {       // Loop over the shared dimension
            for (int j = 0; j < n; j++) {  // Loop over columns of C and B
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}
// {
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             float sum = 0.0f;
//             for (int l = 0; l < k; l++) {
//                 sum += A[i * k + l] * B[l * n + j];
//             }
//             C[i * n + j] = sum;
//         }
//     }
// }

// 4. Validate GPU Results Against CPU Results
void validateResults(float *cpu_C, float *gpu_C, int n)
{
    float maxError = 0.0f;
    for (int i = 0; i < n * n; i++) {
        float error = fabs(cpu_C[i] - gpu_C[i]);
        if (error > maxError) {
            maxError = error;
        }
    }

    float maxErrorThreshHold = 1e-5f;
    printf("Max Error: %f\n", maxError);
    
    if (maxError < maxErrorThreshHold) {
        printf("Results within acceptable tolerance (%.6f).\n", maxErrorThreshHold);
    } 
    else {
        printf("Max error (%.6f) exceeds tolerance!\n", maxError);
    }
}

// 5. GPU Matrix Multiplication
void matrixMultiplyCUDA(float *h_A, float *h_B, float *h_C, int n, int k)
{
    int sizeA = n * k * sizeof(float);
    int sizeB = k * n * sizeof(float);
    int sizeC = n * n * sizeof(float);

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // Copy data from CPU to GPU
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Define block & grid dimensions using BLOCK_SIZE
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch CUDA kernel
    matrixMulSharedCUDA<<<gridDim, blockDim>>>(d_A, d_B, d_C, n, k);
    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Copy result from GPU to CPU
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Compute approximate GFLOPS
    double gpu_gflops = (2.0 * (double)k * (double)n * (double)n) / (gpu_time * 1e6);
    printf("CUDA: n=%d, k=%d, BLOCK_SIZE=%d, Time=%.6f ms, GFLOPS=%.2f\n",
           n, k, BLOCK_SIZE, gpu_time, gpu_gflops);

    // Cleanup GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 6. Serial Matrix Multiplication on the CPU
void matrixMultiplySerial(float *A, float *B, float *C, int n, int k)
{
            // Compute reference CPU multiplication
            clock_t start_cpu = clock();
            matrixMultiplyCPU(A, B, C, n, k);
            clock_t stop_cpu = clock();
            double cpu_time = 1000.0 * (stop_cpu - start_cpu) / CLOCKS_PER_SEC;

            // CPU GFLOPS estimate
            double cpu_gflops = (2.0 * (double)k * (double)n * (double)n) / (cpu_time * 1e6);
            printf("[CPU] Time=%.3f ms, %.2f GFLOPS\n", cpu_time, cpu_gflops);
}

// 7. Main function
int main()
{
    // Example arrays of (n) and (k) to test
    int n_values[] = {512, 1024, 2048, 4096};  
    int k_values[] = {32, 48, 64, 96, 128};   
    int num_n_values = sizeof(n_values) / sizeof(n_values[0]);
    int num_k_values = sizeof(k_values) / sizeof(k_values[0]);

    // Loop over multiple combinations of n, k
    for (int ni = 0; ni < num_n_values; ni++) {
        int n = n_values[ni];

        for (int ki = 0; ki < num_k_values; ki++) {
            int k = k_values[ki];

            printf("\nTesting with n = %d, k = %d, BLOCK_SIZE = %d\n", n, k, BLOCK_SIZE);

            int sizeA = n * k * sizeof(float);
            int sizeB = k * n * sizeof(float);
            int sizeC = n * n * sizeof(float);

            // Allocate memory on CPU
            float *A     = (float*) malloc(sizeA);
            float *B     = (float*) malloc(sizeB);
            float *C     = (float*) malloc(sizeC);
            float *cpu_C = (float*) calloc(n * n , sizeof(float));

            // Initialize matrices with random values
            srand((unsigned int)time(NULL));
            for (int i = 0; i < n * k; i++) {
                A[i] = (float)rand() / RAND_MAX;
            }
            for (int i = 0; i < k * n; i++) {
                B[i] = (float)rand() / RAND_MAX;
            }
            
            // Run serial multiplication
            matrixMultiplySerial(A, B, cpu_C, n, k);

            // Run CUDA multiplication
            matrixMultiplyCUDA(A, B, C, n, k);

            // Compare CPU vs. GPU results for correctness
            validateResults(cpu_C, C, n);

            // Cleanup
            free(A);
            free(B);
            free(C);
            free(cpu_C);
        }
    }

    return 0;
}
