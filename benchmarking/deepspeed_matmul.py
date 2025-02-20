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

    # Use CPU explicitly.
    device = torch.device("cpu")
    N = args.size

    # For this strategy, we assume both matrices A and B are of size N x N.
    # We'll partition the inner dimension (the shared dimension) equally.
    # Process i will work on the chunk corresponding to columns [k_start:k_end] of A
    # and rows [k_start:k_end] of B.
    if rank == 0:
        A = torch.rand(N, N, dtype=dtype, device=device)
        B = torch.rand(N, N, dtype=dtype, device=device)
    else:
        A = torch.empty(N, N, dtype=dtype, device=device)
        B = torch.empty(N, N, dtype=dtype, device=device)

    # Broadcast both matrices A and B to all processes.
    torch.distributed.broadcast(A, src=0)
    torch.distributed.broadcast(B, src=0)

    # Partition the inner dimension (of length N) among world_size processes.
    # For example, if N=10 and world_size=2, then each process gets a chunk of size 5.
    base = N // world_size
    rem = N % world_size
    # Distribute any remainder among the first 'rem' processes.
    k_start = rank * base + (rank if rank < rem else rem)
    k_count = base + (1 if rank < rem else 0)
    k_end = k_start + k_count

    timings = []
    total_iterations = args.count + args.warmup

    for i in range(total_iterations):
        t0 = time.time()
        # Each process computes a partial product:
        #   partial_C = A[:, k_start:k_end] dot B[k_start:k_end, :]
        local_C = torch.matmul(A[:, k_start:k_end], B[k_start:k_end, :])
        # Copy the result to be reduced.
        final_C = local_C.clone()
        # Allreduce (sum) the partial results across all processes.
        if args.ccl:
            dist.all_reduce(final_C)
        else:
            dist.inference_all_reduce(final_C)
        t1 = time.time()
        if i >= args.warmup:
            timings.append(t1 - t0)

    # Compute statistics.
    t_min = min(timings)
    t_max = max(timings)
    t_avg = sum(timings) / len(timings)
    variance = sum((t - t_avg) ** 2 for t in timings) / len(timings)
    stddev = math.sqrt(variance) / t_avg * 100 if t_avg != 0 else 0

    # Only rank 0 writes the CSV header and results.
    if rank == 0:
        header = ["Implementation", "CPU_Count", "Matrix_Size", "Iterations", "t_min(s)", "t_max(s)", "t_avg(s)", "stddev(%)"]
        row = ["DeepSpeed", world_size, N, args.count, f"{t_min:.6f}", f"{t_max:.6f}", f"{t_avg:.6f}", f"{stddev:.2f}"]
        with open(args.outfile, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow(row)

if __name__ == "__main__":
    main()
