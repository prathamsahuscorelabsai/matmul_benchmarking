/* mpi_matmul.c */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char **argv) {
    int N = 1024;       // Default matrix size (NxN)
    int count = 10;     // Number of timed iterations
    int warmup = 2;     // Number of warmup iterations (not measured)
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        count = atoi(argv[2]);
    }
    if (argc > 3) {
        warmup = atoi(argv[3]);
    }
    int total_iterations = count + warmup;

    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Allocate matrices.
    double *A = (double *) malloc(N * N * sizeof(double));
    double *B = (double *) malloc(N * N * sizeof(double));
    double *local_C = (double *) malloc(N * N * sizeof(double));
    double *C = (double *) malloc(N * N * sizeof(double));
    if (A == NULL || B == NULL || local_C == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed on rank %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    // Rank 0 initializes matrices A and B.
    if (rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < N * N; i++){
            A[i] = (double) rand() / RAND_MAX;
            B[i] = (double) rand() / RAND_MAX;
        }
    }
    // Broadcast matrices.
    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Partition the inner (K) dimension among processes.
    int base = N / world_size;
    int rem = N % world_size;
    int k_start = rank * base + (rank < rem ? rank : rem);
    int k_count = base + (rank < rem ? 1 : 0);
    int k_end = k_start + k_count;
    
    double *timings = (double *) malloc(count * sizeof(double));
    if (timings == NULL) {
        fprintf(stderr, "Memory allocation for timings failed on rank %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int measured = 0;
    
    for (int iter = 0; iter < total_iterations; iter++){
        // Reset local_C to zero.
        for (int i = 0; i < N * N; i++){
            local_C[i] = 0.0;
        }
        
        double t0 = MPI_Wtime();
        
        //Is the implementation of torch matmul much more optimised than this which is causing network latency to overexceed this?

        for (int i = 0; i < N; i++){
            for (int k = k_start; k < k_end; k++){
                double a_val = A[i * N + k];
                for (int j = 0; j < N; j++){
                    local_C[i * N + j] += a_val * B[k * N + j];
                }
            }
        }
        
        // Sum up partial results.
        MPI_Allreduce(local_C, C, N * N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        double t1 = MPI_Wtime();
        if (iter >= warmup) {
            timings[measured++] = t1 - t0;
        }
    }
    
    // Compute statistics.
    double t_min = timings[0], t_max = timings[0], sum = 0.0;
    for (int i = 0; i < count; i++){
        if (timings[i] < t_min) t_min = timings[i];
        if (timings[i] > t_max) t_max = timings[i];
        sum += timings[i];
    }
    double t_avg = sum / count;
    double variance = 0.0;
    for (int i = 0; i < count; i++){
        double diff = timings[i] - t_avg;
        variance += diff * diff;
    }
    variance /= count;
    double stddev = (t_avg > 0.0) ? (sqrt(variance) / t_avg * 100) : 0.0;
    
    // Only rank 0 prints the CSV header and result.
    if (rank == 0) {
        printf("Implemendtation,CPU_Count,Matrix_Size,Iterations,t_min(s),t_max(s),t_avg(s),stddev(%%)\n");
        printf("MPI,%d,%d,%d,%.6f,%.6f,%.6f,%.2f\n",
               world_size, N, count, t_min, t_max, t_avg, stddev);
    }
    
    free(A);
    free(B);
    free(local_C);
    free(C);
    free(timings);
    
    MPI_Finalize();
    return 0;
}
