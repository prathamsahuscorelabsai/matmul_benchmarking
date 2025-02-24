deepspeed --num_accelerators=4 --bind_cores_to_rank deepspeed_matmul.py --size 512 --count 10 --warmup 2 --ccl --dtype fp32 --outfile "temp.csv"
# mpirun -n 4 python mpi4py_matmul.py --size 512 --count 10 --warmup 2 --dtype fp32 --outfile "temp.csv" > /dev/null 2>&1
