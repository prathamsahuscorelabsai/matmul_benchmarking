#!/usr/bin/env python3
from mpi4py import MPI
import torch
import numpy as np
import argparse
import math
import csv

def main():
    parser = argparse.ArgumentParser(
        description="Distributed Matrix Multiplication using mpi4py and torch (local multiplication with torch.matmul)"
    )
    parser.add_argument("--size", type=int, default=1024,
                        help="Matrix dimension (NxN)")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"],
                        default="fp32", help="Data type for matrices")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup iterations (not measured)")
    parser.add_argument("--outfile", type=str, default="results_mpi4py.csv",
                        help="CSV output file")
    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Choose torch data type.
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16

    device = torch.device("cpu")
    N = args.size

    # Rank 0 initializes matrices A and B.
    if rank == 0:
        A = torch.rand((N, N), dtype=dtype, device=device)
        B = torch.rand((N, N), dtype=dtype, device=device)
    else:
        A = torch.empty((N, N), dtype=dtype, device=device)
        B = torch.empty((N, N), dtype=dtype, device=device)

    # Broadcast both matrices from rank 0 to all processes.
    # (Note: torch tensors on CPU support the buffer interface.)
    A_np = A.numpy()
    B_np = B.numpy()
    comm.Bcast(A_np, root=0)
    comm.Bcast(B_np, root=0)

    # Partition the inner dimension among processes.
    base = N // world_size
    rem = N % world_size
    if rank < rem:
        k_start = rank * (base + 1)
        k_count = base + 1
    else:
        k_start = rem * (base + 1) + (rank - rem) * base
        k_count = base
    k_end = k_start + k_count

    total_iterations = args.count + args.warmup
    timings = []

    for i in range(total_iterations):
        t0 = MPI.Wtime()
        # Use torch.matmul for local matrix multiplication.
        local_C = torch.matmul(A[:, k_start:k_end], B[k_start:k_end, :])
        # Prepare a numpy array for the reduced result.
        local_C_np = local_C.numpy()
        result_np = np.empty_like(local_C_np)
        # Allreduce (sum) the partial results across all processes.
        comm.Allreduce(local_C_np, result_np, op=MPI.SUM)
        t1 = MPI.Wtime()
        if i >= args.warmup:
            timings.append(t1 - t0)

    if rank == 0:
        t_min = min(timings)
        t_max = max(timings)
        t_avg = sum(timings) / len(timings)
        variance = sum((t - t_avg) ** 2 for t in timings) / len(timings)
        stddev = (math.sqrt(variance) / t_avg * 100) if t_avg > 0 else 0
        header = ["Implementation", "CPU_Count", "Matrix_Size", "Iterations", "t_min(s)", "t_max(s)", "t_avg(s)", "stddev(%)"]
        row = ["mpi4py", world_size, N, args.count,
               f"{t_min:.6f}", f"{t_max:.6f}", f"{t_avg:.6f}", f"{stddev:.2f}"]
        with open(args.outfile, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(row)
        print(",".join(map(str, row)))

    MPI.Finalize()

if __name__ == '__main__':
    main()
