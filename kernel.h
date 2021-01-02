#ifndef __kernel__
#define __kernel__

class Kernel
{
  public:
	Kernel();
	__host__ __device__ void addex(int n, float* x, float* y);
};

#endif //__kernel__
