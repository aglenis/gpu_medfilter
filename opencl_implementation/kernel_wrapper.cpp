#include "../timer.h"

namespace compute = boost::compute;

void medianFilter2D_wrapper(compute::command_queue queue,boost::compute::program foo_program,compute::buffer gpu_in,compute::buffer gpu_out,compute::buffer gpu_histogram,int heightImage,int widthImage,int implementation)
{
  try{
  boost::compute::kernel foo_kernel;
switch(implementation)
{
  case 1:
    std::cout<<"running naive median filter"<<std::endl;
    foo_kernel = foo_program.create_kernel("MedianFilter2D");
    break;
  case 2:
    std::cout<<"running histogram median filter"<<std::endl;
    foo_kernel = foo_program.create_kernel("MedianFilter2D_histogram");
    break;  
  case 3:
    std::cout<<"running median filter with partial selection"<<std::endl;
    foo_kernel = foo_program.create_kernel("MedianFilter2D_partial");
    break;
  case 4:
    std::cout<<"running median filter with forgetful selection"<<std::endl;
   foo_kernel = foo_program.create_kernel("MedianFilter2D_forgetful");
    break;
  case 5:
    std::cout<<"running median filter with fast histogram"<<std::endl;
   foo_kernel = foo_program.create_kernel("histogram2d");
    break;
  
}   

if(implementation!=5)
{
         // TODO these are the arguments for the first kernel
    foo_kernel.set_arg(0,gpu_in);
    foo_kernel.set_arg(1,gpu_out);
    foo_kernel.set_arg(2,sizeof(int),&widthImage);
    foo_kernel.set_arg(3,sizeof(int),&heightImage);
//     foo_kernel.set_arg(4,sizeof(unsigned int),&window_size);

        // Launch kernel

   
   const size_t offset[] = { 0, 0 };
   const size_t bounds[] = { heightImage, widthImage };
    timer kernel_timer1;
    queue.enqueue_nd_range_kernel(foo_kernel, 2, 
                              offset, 
                              bounds, 
                              0);
        double time_elapsed1=kernel_timer1.elapsed();
    printf("total time elapsed for the kernel implementation %d is %f \n",implementation,time_elapsed1);
}
else
{
      foo_kernel.set_arg(0,gpu_in);
    foo_kernel.set_arg(1,gpu_out);
    foo_kernel.set_arg(2,gpu_histogram);
    foo_kernel.set_arg(3,sizeof(int),&widthImage);
    foo_kernel.set_arg(4,sizeof(int),&heightImage);
//     foo_kernel.set_arg(4,sizeof(unsigned int),&window_size);

        // Launch kernel 
    timer kernel_timer;
    queue.enqueue_1d_range_kernel(foo_kernel, 0,heightImage, 0);
    double time_elapsed=kernel_timer.elapsed();
    printf("total time elapsed for the kernel implementation %d is %f \n",implementation,time_elapsed);
}
  }
  catch(boost::compute::opencl_error &e){
  
    std::cout<<"something went wrong with kernel execution"<<std::endl;
  }
}