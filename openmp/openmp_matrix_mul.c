// 1. Headers
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>

// 2. Transpose Matrix for Cache Efficiency
void transposeMatrix(float *B, float *B_T, int k, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B_T[j * k + i] = B[i * n + j];  // Transpose B
        }
    }
}

// 3. Parallel Matrix Multiplication Using OpenMP + SIMD
void matrixMultiplyOpenMPKernel(float *A, float *B_T, float *C, int n, int k)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;

            // SIMD to speed up inner loop
            #pragma omp simd aligned(A, B_T: 64) reduction(+:sum)  
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B_T[j * k + l];  
            }

            C[i * n + j] = sum;
        }
    }
}

// 4. Serial CPU Matrix Multiplication
void matrixMultiplySerial(float *A, float *B, float *C, int n, int k)
{
    clock_t start_cpu = clock();
    
    for (int i = 0; i < n; i++) {           // Loop over rows of A and C 
        for (int l = 0; l < k; l++) {       // Loop over the shared dimension
            for (int j = 0; j < n; j++) {  // Loop over columns of C and B
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
    
    clock_t end_cpu = clock();
    double cpu_time = 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC;
    double cpu_gflops = (2.0 * k * n * n) / (cpu_time * 1e6);

    printf("[CPU] n=%d, k=%d, Time=%.6f ms, GFLOPS=%.2f\n", n, k, cpu_time, cpu_gflops);
}


// 5. OpenMP Matrix Multiplication
void matrixMultiplyOpenMP(float *A, float *B_T, float *C, int n, int k)
{
    double start_omp = omp_get_wtime();

    // Perform OpenMP matrix multiplication
    matrixMultiplyOpenMPKernel(A, B_T, C, n, k);

    double end_omp = omp_get_wtime();
    double omp_time = (end_omp - start_omp) * 1000.0; // Convert to milliseconds

    // Compute OpenMP GFLOPS estimate
    double omp_gflops = (2.0 * (double)k * (double)n * (double)n) / (omp_time * 1e6);
    printf("[OpenMP] n=%d, k=%d, Threads=%d, Time=%.6f ms, GFLOPS=%.2f\n",
           n, k, omp_get_max_threads(), omp_time, omp_gflops);
}

// 6. Validate OpenMP Results Against CPU Results
void validateResults(float *cpu_C, float *omp_C, int n)
{
    float maxError = 0.0f;
    for (int i = 0; i < n * n; i++) {
        float error = fabs(cpu_C[i] - omp_C[i]);
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


// 7. Main Function for Testing OpenMP Matrix Multiplication
int main(void)
{
    int n_values[] = {512, 1024, 2048, 4096};
    int k_values[] = {32, 48, 64, 96, 128};
    int num_n_values = sizeof(n_values) / sizeof(n_values[0]);
    int num_k_values = sizeof(k_values) / sizeof(k_values[0]);

    for (int ni = 0; ni < num_n_values; ni++) {
        int n = n_values[ni];

        for (int ki = 0; ki < num_k_values; ki++) {
            int k = k_values[ki];

            printf("\nTesting with n = %d, k = %d, %d threads\n", n, k, omp_get_max_threads());

            int sizeA = n * k * sizeof(float);
            int sizeB = k * n * sizeof(float);
            int sizeC = n * n * sizeof(float);

            // Allocate memory on CPU
            float *A     = (float*) malloc(sizeA);
            float *B     = (float*) malloc(sizeB);
            float *B_T   = (float*) malloc(sizeB);
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

            // Transpose B for better memory access
            transposeMatrix(B, B_T, k, n);

            // Run serial CPU multiplication
            matrixMultiplySerial(A, B, cpu_C, n, k);

            // Run OpenMP multiplication
            matrixMultiplyOpenMP(A, B_T, C, n, k);

            // Validate results
            validateResults(cpu_C, C, n);

            // Cleanup
            free(A);
            free(B);
            free(B_T);
            free(C);
            free(cpu_C);
        }
    }

    return 0;
}
