# README

## Overview
This project involves running parallel matrix multiplication using CUDA and job submission using PBS. The following instructions guide you through running the necessary scripts for the setup and execution of the tasks.

## Prerequisites
- Ensure that you have access to an HPC system with CUDA support.

## Running the Scripts

### Step 1: Follow cuda_2024-3.pdf to set up env

This will set up the environment for the OpenMP matrix multiplication and ensure that the necessary configurations are in place.

### Step 2: Copy 03-matrix_mul.cu onto Cresent system (make sure you are in the dir where 03-matrix_mul.cu is at)
scp 03-matrix_mul.cu  duong.vu@crescent2.central.cranfield.ac.uk:~/

1. Once `03-matrix_mul.cu` has copy completed successfully, you can follow the pdf file and submit your job.
