#include "kernel.h"

Kernel::Kernel()
{
}

__host__ __device__ void Kernel::addex(int n, float* x, float* y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

