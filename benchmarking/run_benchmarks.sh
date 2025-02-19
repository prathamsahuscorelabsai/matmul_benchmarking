#!/bin/bash
# This script runs the DeepSpeed and MPI implementations for multiple
# matrix sizes and CPU counts, and aggregates the CSV results to separate files.

# Define matrix sizes and CPU counts.
matrix_sizes=(256 512 1024 2048 4096 8192 16384)
cpu_counts=(1 2 4 8)

# Final output CSV file names.
deepspeed_csv="deepspeed_results.csv"
mpi_csv="mpi_results.csv"

# Write CSV headers.
echo "Implementation,CPU_Count,Matrix_Size,Iterations,t_min(s),t_max(s),t_avg(s),stddev(%)" > $deepspeed_csv
echo "Implementation,CPU_Count,Matrix_Size,Iterations,t_min(s),t_max(s),t_avg(s),stddev(%)" > $mpi_csv

# Temporary file names.
temp_ds="temp_ds.csv"
temp_mpi="temp_mpi.csv"

# Run DeepSpeed benchmarks.
for size in "${matrix_sizes[@]}"; do
    for cpus in "${cpu_counts[@]}"; do
        echo "Running DeepSpeed: Matrix size $size, CPUs $cpus"
        # Run the DeepSpeed benchmark (assuming the updated file is named deepspeed_matmul.py)
        # and write its CSV output to a temporary file.
        deepspeed --num_accelerators=$cpus deepspeed_matmul.py --size $size --count 10 --warmup 2 --ccl --dtype fp32 --outfile $temp_ds > /dev/null 2>&1
        # Append the result row (skip the header) to the master CSV.
        tail -n +2 "$temp_ds" >> "$deepspeed_csv"
        rm -f "$temp_ds"
    done
done

# Run MPI benchmarks.
for size in "${matrix_sizes[@]}"; do
    for cpus in "${cpu_counts[@]}"; do
        echo "Running MPI: Matrix size $size, CPUs $cpus"
        # Run the MPI benchmark (assuming your MPI binary is named mpi_matmul)
        # and redirect its CSV output to a temporary file.
        mpirun -n $cpus ./mpi_matmul $size 10 2 > "$temp_mpi" 2>/dev/null
        # Append the result row (skip the header) to the master CSV.
        tail -n +2 "$temp_mpi" >> "$mpi_csv"
        rm -f "$temp_mpi"
    done
done

echo "Benchmarking complete. Results are in $deepspeed_csv and $mpi_csv."
