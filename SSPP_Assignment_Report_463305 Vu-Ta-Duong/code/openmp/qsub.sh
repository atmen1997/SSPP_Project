gcc -fopenmp -O4 -o openmp_matrix_mul openmp_matrix_mul.c
qsub -V openmp_add.sub

