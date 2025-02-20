#!/bin/bash
# This script runs the MPI (C implementation) benchmark for multiple
# matrix sizes and CPU counts, and aggregates the CSV results into a single file.

# Define matrix sizes and CPU counts.
matrix_sizes=(256 512 1024 2048 4096 8192 16384)
cpu_counts=(1 2 4 8)

# Final output CSV file name.
mpi_csv="mpi_results.csv"

# Write CSV header.
echo "Implementation,CPU_Count,Matrix_Size,Iterations,t_min(s),t_max(s),t_avg(s),stddev(%)" > "$mpi_csv"

# Temporary file for CSV output.
temp_mpi="temp_mpi.csv"

# Loop over matrix sizes and CPU counts.
for size in "${matrix_sizes[@]}"; do
    for cpus in "${cpu_counts[@]}"; do
        echo "Running MPI (C): Matrix size $size, CPUs $cpus"
        # Run the MPI benchmark (assumes your MPI binary is named mpi_matmul)
        mpirun -n $cpus ./mpi_matmul $size 10 2 > "$temp_mpi"
        # Append the result row (skip the header) to the master CSV.
        tail -n +2 "$temp_mpi" >> "$mpi_csv"
        rm -f "$temp_mpi"
    done
done

echo "MPI (C) benchmarking complete. Results are in $mpi_csv."
