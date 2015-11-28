#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
 
 
void MedianFilterUcharCUDA(unsigned char *input,unsigned char* output,unsigned int * d_histogram,int widthImage, int heightImage,int window_size, int threadsX, int threadsY,int implementation);