shared memory :
变量前加 __shared__ It creates a copy of the variable for each block
that you launch on the GPU. Every thread in that block shares the memory, but
threads cannot see or modify the copy of this variable that is seen within other
blocks.But nothing in life is free, and interthread communication is no exception.
If we expect to communicate between threads, we also need a mechanism for
synchronizing between threads

synchronize:
__syncthreads();
This call guarantees that every thread in the block has completed instructions
prior to the __syncthreads() before the hardware will execute the next instruction on any thread

