#!/bin/bash
# This script runs the MPI (mpi4py implementation) benchmark for multiple
# matrix sizes and CPU counts, and aggregates the CSV results into a single file.

# Define matrix sizes and CPU counts.
matrix_sizes=(256 512 1024 2048 4096 8192 16384)
cpu_counts=(1 2 4 8)

# Final output CSV file name.
mpi4py_csv="mpi4py_results.csv"

# Write CSV header.
#  header = [
#             "Implementation", "CPU_Count", "Matrix_Size", "Iterations",
#             "Matmul_t_min(s)", "Matmul_t_max(s)", "Matmul_t_avg(s)", "Matmul_stddev(%)",
#             "Allreduce_t_min(s)", "Allreduce_t_max(s)", "Allreduce_t_avg(s)", "Allreduce_stddev(%)",
#             "Total_t_min(s)", "Total_t_max(s)", "Total_t_avg(s)", "Total_stddev(%)"
#         ]

echo "Implementation,CPU_Count,Matrix_Size,Iterations,Matmul_t_min(s),Matmul_t_max(s),Matmul_t_avg(s),Matmul_stddev(%),Allreduce_t_min(s),Allreduce_t_max(s),Allreduce_t_avg(s),Allreduce_stddev(%),Total_t_min(s),Total_t_max(s),Total_t_avg(s),Total_stddev(%)" > "$mpi4py_csv"


# Temporary file for CSV output.
temp_mpi4py="temp_mpi4py.csv"

# Loop over matrix sizes and CPU counts.
for size in "${matrix_sizes[@]}"; do
    for cpus in "${cpu_counts[@]}"; do
        echo "Running MPI (mpi4py): Matrix size $size, CPUs $cpus"
        # Run the mpi4py benchmark (assumes your file is named mpi4py_matmul.py)
        mpirun -n $cpus python mpi4py_matmul.py --size $size --count 10 --warmup 2 --dtype fp32 --outfile "$temp_mpi4py" > /dev/null 2>&1
        # Append the result row (skip the header) to the master CSV.
        tail -n +2 "$temp_mpi4py" >> "$mpi4py_csv"
        rm -f "$temp_mpi4py"
    done
done

echo "MPI (mpi4py) benchmarking complete. Results are in $mpi4py_csv."
