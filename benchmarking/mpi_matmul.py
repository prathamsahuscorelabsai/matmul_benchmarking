#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import argparse

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="MPI Matrix Multiplication")
    parser.add_argument("--size", type=int, default=1000, help="Matrix size (NxN)")
    args = parser.parse_args()
    N = args.size

    # Process 0 initializes matrices; others prepare empty arrays.
    if rank == 0:
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
    else:
        A = np.empty((N, N), dtype='d')
        B = np.empty((N, N), dtype='d')
    
    # Broadcast matrices A and B to all processes.
    comm.Bcast(A, root=0)
    comm.Bcast(B, root=0)

    # Partition the inner dimension (summation index) among processes.
    # Create a list of counts that divides N as evenly as possible.
    counts = [(N // size) + (1 if i < (N % size) else 0) for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]
    k_start = displs[rank]
    k_end = k_start + counts[rank]

    # Each process computes its partial contribution:
    # C_partial = A[:, k_start:k_end] dot B[k_start:k_end, :]
    local_C = np.dot(A[:, k_start:k_end], B[k_start:k_end, :])

    # Allocate space for the final result.
    C = np.empty((N, N), dtype='d')
    # Allreduce to sum up the partial results from all processes.
    comm.Allreduce(local_C, C, op=MPI.SUM)

    if rank == 0:
        print("Result matrix shape:", C.shape)

if __name__ == "__main__":
    main()
