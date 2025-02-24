#!/usr/bin/env python3
from mpi4py import MPI
import torch
import numpy as np
import argparse
import math
import csv
import time

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

    # Initialize MPI.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Choose the torch data type.
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
    # A_np = A.numpy()
    # B_np = B.numpy()
    comm.Bcast(A, root=0)
    comm.Bcast(B, root=0)

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
    timings_matmul = []
    timings_allreduce = []
    final_result_np = None

    for i in range(total_iterations):
        # Measure local matmul time.
        t0 = time.time()
        local_C = torch.matmul(A[:, k_start:k_end], B[k_start:k_end, :])
        t1 = time.time()

        # Prepare a numpy array for the reduced result.
        # local_C_np = local_C.numpy()
        # result_np = np.empty_like(local_C_np)

        result = torch.empty((N, N), dtype=dtype, device=device)
        # Allreduce (sum) the partial results across all processes.

        t2 = time.time()
        comm.Allreduce(local_C, result, op=MPI.SUM)
        t3 = time.time()

        # Save the last iteration result for validation.
        final_result_np = result.clone()

        # Only record timings after warmup iterations.
        if i >= args.warmup:
            timings_matmul.append(t1 - t0)
            timings_allreduce.append(t3 - t2)

    if rank == 0:
        # Validate the distributed result by comparing with single-core multiplication.
        C_single = torch.matmul(A, B)
        # C_single_np = C_single.numpy()
        if not torch.allclose(final_result_np, C_single, rtol=1e-3, atol=1e-5):
            print("Result check FAILED: Distributed result does not match single-core matmul result!")
            return
        else:
            print("Result check passed: Distributed result matches single-core matmul result.")

        # Compute statistics for local matmul timings.
        matmul_t_min = min(timings_matmul)
        matmul_t_max = max(timings_matmul)
        matmul_t_avg = sum(timings_matmul) / len(timings_matmul)
        var_matmul = sum((t - matmul_t_avg) ** 2 for t in timings_matmul) / len(timings_matmul)
        stddev_matmul = math.sqrt(var_matmul) / matmul_t_avg * 100 if matmul_t_avg != 0 else 0

        # Compute statistics for allreduce timings.
        allreduce_t_min = min(timings_allreduce)
        allreduce_t_max = max(timings_allreduce)
        allreduce_t_avg = sum(timings_allreduce) / len(timings_allreduce)
        var_allreduce = sum((t - allreduce_t_avg) ** 2 for t in timings_allreduce) / len(timings_allreduce)
        stddev_allreduce = math.sqrt(var_allreduce) / allreduce_t_avg * 100 if allreduce_t_avg != 0 else 0

        # Compute total time per iteration.
        total_times = [m + a for m, a in zip(timings_matmul, timings_allreduce)]
        total_t_min = min(total_times)
        total_t_max = max(total_times)
        total_t_avg = sum(total_times) / len(total_times)
        var_total = sum((t - total_t_avg) ** 2 for t in total_times) / len(total_times)
        stddev_total = math.sqrt(var_total) / total_t_avg * 100 if total_t_avg != 0 else 0

        print(f"Total time metrics: min={total_t_min:.6f}s, max={total_t_max:.6f}s, avg={total_t_avg:.6f}s, stddev={stddev_total:.2f}%")

        header = [
            "Implementation", "CPU_Count", "Matrix_Size", "Iterations",
            "Matmul_t_min(s)", "Matmul_t_max(s)", "Matmul_t_avg(s)", "Matmul_stddev(%)",
            "Allreduce_t_min(s)", "Allreduce_t_max(s)", "Allreduce_t_avg(s)", "Allreduce_stddev(%)",
            "Total_t_min(s)", "Total_t_max(s)", "Total_t_avg(s)", "Total_stddev(%)"
        ]
        row = [
            "mpi4py", world_size, N, args.count,
            f"{matmul_t_min:.6f}", f"{matmul_t_max:.6f}", f"{matmul_t_avg:.6f}", f"{stddev_matmul:.2f}",
            f"{allreduce_t_min:.6f}", f"{allreduce_t_max:.6f}", f"{allreduce_t_avg:.6f}", f"{stddev_allreduce:.2f}",
            f"{total_t_min:.6f}", f"{total_t_max:.6f}", f"{total_t_avg:.6f}", f"{stddev_total:.2f}"
        ]
        with open(args.outfile, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(row)
        print(",".join(map(str, row)))

    MPI.Finalize()

if __name__ == '__main__':
    main()
