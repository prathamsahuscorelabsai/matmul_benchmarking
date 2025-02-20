#!/bin/bash
# This script runs the DeepSpeed implementation benchmark for multiple
# matrix sizes and CPU counts, and aggregates the CSV results into a single file.

# Define matrix sizes and CPU counts.
matrix_sizes=(256 512 1024 2048 4096 8192 16384)
cpu_counts=(1 2 4 8)

# Final output CSV file name.
deepspeed_csv="deepspeed_results.csv"

# Write CSV header.
echo "Implementation,CPU_Count,Matrix_Size,Iterations,t_min(s),t_max(s),t_avg(s),stddev(%)" > "$deepspeed_csv"

# Temporary file for CSV output.
temp_ds="temp_ds.csv"

# Loop over matrix sizes and CPU counts.
for size in "${matrix_sizes[@]}"; do
    for cpus in "${cpu_counts[@]}"; do
        echo "Running DeepSpeed: Matrix size $size, CPUs $cpus"
        # Run the DeepSpeed benchmark (assumes your file is named deepspeed_matmul.py)
        deepspeed --num_accelerators=$cpus deepspeed_matmul.py --size $size --count 10 --warmup 2 --ccl --dtype fp32 --outfile "$temp_ds" > /dev/null 2>&1
        # Append the result row (skip the header) to the master CSV.
        tail -n +2 "$temp_ds" >> "$deepspeed_csv"
        rm -f "$temp_ds"
    done
done

echo "DeepSpeed benchmarking complete. Results are in $deepspeed_csv."
