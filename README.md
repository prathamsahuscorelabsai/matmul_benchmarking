# matmul_benchmarking
First cd into the benchmarking directory
```
cd benchmarking
```

## Running the deepspeed code
```
deepspeed --num_accelerators=4 deepspeed_matmul.py --size 1024 --count 10 --warmup 2 --ccl --dtype fp32 > ds_results.csv 2>/dev/null

```

## Compiling the MPI Code and example run
```
mpicc -o mpi_matmul mpi_matmul.c -lm
mpirun -n 4 ./mpi_matmul 1024 10 2 > mpi_results.csv
```

## Running the benchmark

```
./run_benchmarks.sh
```