#!/bin/bash
# This script runs the DeepSpeed implementation benchmark for multiple
# matrix sizes and CPU counts, and aggregates the CSV results into a single file.

# Define matrix sizes and CPU counts.
matrix_sizes=(256 512 1024 2048 4096 8192 16384)
cpu_counts=(1 2 4 8)

# Final output CSV file name.
deepspeed_csv="deepspeed_results_seperated.csv"

# Write CSV header.
# echo "Implementation,CPU_Count,Matrix_Size,Iterations,Matmul_t_min(s),Matmul_t_max(s),Matmul_t_avg(s),Matmul_stddev(%),Allreduce_t_min(s),Allreduce_t_max(s),Allreduce_t_avg(s),Allreduce_stddev(%)" > "$deepspeed_csv"
#  header = [
#             "Implementation", "CPU_Count", "Matrix_Size", "Iterations",
#             "Matmul_t_min(s)", "Matmul_t_max(s)", "Matmul_t_avg(s)", "Matmul_stddev(%)",
#             "Allreduce_t_min(s)", "Allreduce_t_max(s)", "Allreduce_t_avg(s)", "Allreduce_stddev(%)",
#             "Total_t_min(s)", "Total_t_max(s)", "Total_t_avg(s)", "Total_stddev(%)"
#         ]
echo "Implementation,CPU_Count,Matrix_Size,Iterations,Matmul_t_min(s),Matmul_t_max(s),Matmul_t_avg(s),Matmul_stddev(%),Allreduce_t_min(s),Allreduce_t_max(s),Allreduce_t_avg(s),Allreduce_stddev(%),Total_t_min(s),Total_t_max(s),Total_t_avg(s),Total_stddev(%)" > "$deepspeed_csv"
# Temporary file for CSV output.
temp_ds="temp_ds.csv"

# Loop over matrix sizes and CPU counts.
for size in "${matrix_sizes[@]}"; do
    for cpus in "${cpu_counts[@]}"; do
        echo "Running DeepSpeed: Matrix size $size, CPUs $cpus"
        # Run the DeepSpeed benchmark with additional options:
        # --force_multi forces multi-process mode,
        # --launcher impi uses the IPMPI launcher,
        # --bind_cores_to_rank binds cores to each rank.
        deepspeed --num_accelerators=$cpus --bind_cores_to_rank deepspeed_matmul.py --size $size --count 10 --warmup 2 --ccl --dtype fp32 --outfile "$temp_ds" > /dev/null 2>&1

        # Check if the temporary CSV file was created.
        if [ -f "$temp_ds" ]; then
            # Append the result row (skip the header) to the master CSV.
            tail -n +2 "$temp_ds" >> "$deepspeed_csv"
            rm -f "$temp_ds"
        else
            echo "Warning: $temp_ds not found for Matrix size $size, CPUs $cpus" >&2
        fi
    done
done

echo "DeepSpeed benchmarking complete. Results are in $deepspeed_csv."
