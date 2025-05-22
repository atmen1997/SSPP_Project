#include <time.h>
#include <stdlib.h>
#include <stdio.h>

//  Matrix Multiplication Implementations

void multiply_matrix_ijl(float* A, float* B, float* C, int n, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void multiply_matrix_ilj(float* A, float* B, float* C, int n, int k) {
    for (int i = 0; i < n; i++) {
        for (int l = 0; l < k; l++) {
            for (int j = 0; j < n; j++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void multiply_matrix_jil(float* A, float* B, float* C, int n, int k) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void multiply_matrix_jli(float* A, float* B, float* C, int n, int k) {
    for (int j = 0; j < n; j++) {
        for (int l = 0; l < k; l++) {
            for (int i = 0; i < n; i++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void multiply_matrix_lij(float* A, float* B, float* C, int n, int k) {
    for (int l = 0; l < k; l++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void multiply_matrix_lji(float* A, float* B, float* C, int n, int k) {
    for (int l = 0; l < k; l++) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

//  Main Function

int main() {
    int n_values[] = {512, 1024, 2048, 4096};  
    int k_values[] = {32, 48, 64, 96, 128};
    int num_n_values = sizeof(n_values) / sizeof(n_values[0]);
    int num_k_values = sizeof(k_values) / sizeof(k_values[0]);

    srand(time(NULL));

    for (int ni = 0; ni < num_n_values; ni++) {
        int n = n_values[ni];

        for (int ki = 0; ki < num_k_values; ki++) {
            int k = k_values[ki];

            printf("\nTesting n = %d, k = %d\n", n, k);

            float *A = (float*) malloc(n * k * sizeof(float));
            float *B = (float*) malloc(k * n * sizeof(float));
            float *C = (float*) calloc(n * n, sizeof(float)); 

            if (!A || !B || !C) {
                printf("Memory allocation failed!\n");
                return 1;
            }

            // Initialize random matrices
            for (int i = 0; i < n * k; i++) A[i] = (float)rand() / RAND_MAX;
            for (int i = 0; i < k * n; i++) B[i] = (float)rand() / RAND_MAX;

            // Run Each Multiplication Order

            clock_t start, end;
            double time_taken, gflops;

            // I-J-L Order
            start = clock();
            multiply_matrix_ijl(A, B, C, n, k);
            end = clock();
            time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
            gflops = (2.0 * k * n * n) / (time_taken * 1e9);
            printf("I-J-L: Time = %.6f sec, GFLOPS = %.2f\n", time_taken, gflops);

            // I-L-J Order
            start = clock();
            multiply_matrix_ilj(A, B, C, n, k);
            end = clock();
            time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
            gflops = (2.0 * k * n * n) / (time_taken * 1e9);
            printf("I-L-J: Time = %.6f sec, GFLOPS = %.2f\n", time_taken, gflops);

            // J-I-L Order
            start = clock();
            multiply_matrix_jil(A, B, C, n, k);
            end = clock();
            time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
            gflops = (2.0 * k * n * n) / (time_taken * 1e9);
            printf("J-I-L: Time = %.6f sec, GFLOPS = %.2f\n", time_taken, gflops);

            // J-L-I Order
            start = clock();
            multiply_matrix_jli(A, B, C, n, k);
            end = clock();
            time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
            gflops = (2.0 * k * n * n) / (time_taken * 1e9);
            printf("J-L-I: Time = %.6f sec, GFLOPS = %.2f\n", time_taken, gflops);

            // L-I-J Order 
            start = clock();
            multiply_matrix_lij(A, B, C, n, k);
            end = clock();
            time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
            gflops = (2.0 * k * n * n) / (time_taken * 1e9);
            printf("L-I-J: Time = %.6f sec, GFLOPS = %.2f\n", time_taken, gflops);

            // L-J-I Order 
            start = clock();
            multiply_matrix_lji(A, B, C, n, k);
            end = clock();
            time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
            gflops = (2.0 * k * n * n) / (time_taken * 1e9);
            printf("L-J-I: Time = %.6f sec, GFLOPS = %.2f\n", time_taken, gflops);

            free(A);
            free(B);
            free(C);
        }
    }

    return 0;
}
