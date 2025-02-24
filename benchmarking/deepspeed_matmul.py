#!/usr/bin/env python3
import os
# Set environment variables before importing DeepSpeed to reduce logging.
os.environ["DS_VERBOSE"] = "0"
os.environ["DEEPSPEED_LOG_LEVEL"] = "error"

import logging
# Set the root logger and DeepSpeed’s logger to only show ERROR messages.
logging.basicConfig(level=logging.ERROR)
logging.getLogger("deepspeed").setLevel(logging.ERROR)

import argparse
import time
import math
import csv
import torch
import deepspeed
import deepspeed.comm as dist

def main():
    parser = argparse.ArgumentParser(
        description="Distributed Matrix Multiplication using DeepSpeed on CPU with inner-dimension partitioning"
    )
    # Accept local_rank so that DeepSpeed launcher doesn't complain.
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank passed by DeepSpeed launcher")
    parser.add_argument("--size", type=int, default=1024,
                        help="Matrix dimension (NxN)")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"],
                        default="fp32", help="Data type for matrices")
    parser.add_argument("--ccl", action="store_true",
                        help="Use dist.all_reduce (ccl) instead of dist.inference_all_reduce")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup iterations (not measured)")
    parser.add_argument("--outfile", type=str, default="results.csv",
                        help="CSV output file")
    args = parser.parse_args()

    # Choose the torch data type.
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16

    # Initialize DeepSpeed distributed.
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # print(f"Rank {rank}/{world_size} reporting for duty!")
    # print(f"Rank {rank} is running on core {os.sched_getaffinity(0)}")

    # Use CPU explicitly.
    device = torch.device("cpu")
    N = args.size

    # For this strategy, we assume both matrices A and B are of size N x N.
    if rank == 0:
        A = torch.rand(N, N, dtype=dtype, device=device)
        B = torch.rand(N, N, dtype=dtype, device=device)
    else:
        A = torch.empty(N, N, dtype=dtype, device=device)
        B = torch.empty(N, N, dtype=dtype, device=device)

    # Broadcast both matrices A and B to all processes.
    torch.distributed.broadcast(A, src=0)
    torch.distributed.broadcast(B, src=0)

    # Partition the inner dimension among world_size processes.
    base = N // world_size
    rem = N % world_size
    if rank < rem:
        k_start = rank * (base + 1)
        k_count = base + 1
    else:
        k_start = rem * (base + 1) + (rank - rem) * base
        k_count = base
    k_end = k_start + k_count

    timings_matmul = []
    timings_allreduce = []
    total_iterations = args.count + args.warmup
    final_C = None  # to hold the last computed distributed result

    # #print size of A and B
    # print(f"Size of A: {A.size()}")
    # print(f"Size of B: {B.size()}")

    for i in range(total_iterations):
        # Measure the matmul section.
        t0 = time.time()
        local_C = torch.matmul(A[:, k_start:k_end], B[k_start:k_end, :])
        t1 = time.time()
        #print rank and size of local_C
        # print(f"Rank {rank} size of local_C: {local_C.size()} and k_start: {k_start} k_end: {k_end}")
        # Copy the result for allreduce.
        final_C = local_C.clone()
        # Measure the all_reduce section.
        t2 = time.time()
        if args.ccl:
            dist.all_reduce(final_C)
        else:
            dist.inference_all_reduce(final_C)  # own implementation
        t3 = time.time()

        # Only record timings after warmup iterations.
        if i >= args.warmup:
            timings_matmul.append(t1 - t0)
            timings_allreduce.append(t3 - t2)

    # Compute statistics for matmul timing.
    matmul_t_min = min(timings_matmul)
    matmul_t_max = max(timings_matmul)
    matmul_t_avg = sum(timings_matmul) / len(timings_matmul)
    var_matmul = sum((t - matmul_t_avg) ** 2 for t in timings_matmul) / len(timings_matmul)
    stddev_matmul = math.sqrt(var_matmul) / matmul_t_avg * 100 if matmul_t_avg != 0 else 0

    # Compute statistics for all_reduce timing.
    allreduce_t_min = min(timings_allreduce)
    allreduce_t_max = max(timings_allreduce)
    allreduce_t_avg = sum(timings_allreduce) / len(timings_allreduce)
    var_allreduce = sum((t - allreduce_t_avg) ** 2 for t in timings_allreduce) / len(timings_allreduce)
    stddev_allreduce = math.sqrt(var_allreduce) / allreduce_t_avg * 100 if allreduce_t_avg != 0 else 0

    # Compute total time per iteration (matmul + allreduce).
    total_times = [m + a for m, a in zip(timings_matmul, timings_allreduce)]
    total_t_min = min(total_times)
    total_t_max = max(total_times)
    total_t_avg = sum(total_times) / len(total_times)
    var_total = sum((t - total_t_avg) ** 2 for t in total_times) / len(total_times)
    stddev_total = math.sqrt(var_total) / total_t_avg * 100 if total_t_avg != 0 else 0

    # On rank 0, check the validity of the distributed result by comparing with single-core computation.
    if rank == 0:
        C_single = torch.matmul(A, B)
        if not torch.allclose(final_C, C_single, rtol=1e-3, atol=1e-5):
            print("Result check FAILED: Distributed result does not match single-core matmul result!")
            return
        else:
            print("Result check passed: Distributed result matches single-core matmul result.")

        # Print total time metrics.
        print(f"Total time metrics: min={total_t_min:.6f}s, max={total_t_max:.6f}s, avg={total_t_avg:.6f}s, stddev={stddev_total:.2f}%")

        # Write results to CSV with separate timing columns.
        header = [
            "Implementation", "CPU_Count", "Matrix_Size", "Iterations",
            "Matmul_t_min(s)", "Matmul_t_max(s)", "Matmul_t_avg(s)", "Matmul_stddev(%)",
            "Allreduce_t_min(s)", "Allreduce_t_max(s)", "Allreduce_t_avg(s)", "Allreduce_stddev(%)",
            "Total_t_min(s)", "Total_t_max(s)", "Total_t_avg(s)", "Total_stddev(%)"
        ]
        row = [
            "DeepSpeed", world_size, N, args.count,
            f"{matmul_t_min:.6f}", f"{matmul_t_max:.6f}", f"{matmul_t_avg:.6f}", f"{stddev_matmul:.2f}",
            f"{allreduce_t_min:.6f}", f"{allreduce_t_max:.6f}", f"{allreduce_t_avg:.6f}", f"{stddev_allreduce:.2f}",
            f"{total_t_min:.6f}", f"{total_t_max:.6f}", f"{total_t_avg:.6f}", f"{stddev_total:.2f}"
        ]
        with open(args.outfile, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(row)

if __name__ == "__main__":
    main()
