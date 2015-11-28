#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
// #include ../timer.h
#include <boost/compute/core.hpp>
#include <boost/compute/source.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "cl_kernels.cl" 
#include "kernel_wrapper.cpp"


void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    if(from.empty())
        return;
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}

int main(int argc, char** argv)
{

	float fTotalTime = 0;

	unsigned int window_size=atoi(argv[2]);
	std::cout<<"running with window size "<<window_size<<std::endl;
	unsigned int total_array_size=window_size*window_size;
	std::string from ="ARRAY_SIZE_ARG";
	std::stringstream ss;
	ss << total_array_size;
	std::string to = ss.str();
	replaceAll(source,from, to);
	std::string from2 ="WINDOW_SIZE";
	std::stringstream ss2;
	ss2 << window_size;
	std::string to2 = ss2.str();
	replaceAll(source,from2, to2);
	std::cout<<source<<std::endl;
	
		  // get the default compute device
    compute::device gpu = compute::system::default_device();

    // create a compute context and command queue
    
    
	  compute::context ctx(gpu);
	  compute::command_queue queue(ctx, gpu);
	    boost::compute::program foo_program =
    boost::compute::program::create_with_source(source, ctx);
    // build the program
try {
    // attempt to compile to program
    foo_program.build();
}
catch(boost::compute::opencl_error &e){
    // program failed to compile, print out the build log
    std::cout << foo_program.build_log() << std::endl;
}
//     boost::compute::kernel foo_kernel = foo_program.create_kernel("MedianFilter2D_forgetful");

	IplImage* gray_image = cvLoadImage(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
	int widthImage=gray_image->width;
	int heightImage=gray_image->height;
	
	IplImage *output_image = cvCreateImage(cvSize(widthImage,heightImage), IPL_DEPTH_8U, 1);
	for( int i=0;i<heightImage;i++)
	  for( int j=0;j<widthImage;j++)
	    output_image->imageData[i*widthImage+j]=255;
	  

    unsigned int size_buffers=heightImage*widthImage*sizeof(unsigned char);
    compute::buffer gpu_in(ctx, size_buffers);
    compute::buffer gpu_out(ctx,size_buffers);
    compute::buffer gpu_histogram(ctx,256*heightImage*sizeof(unsigned int));
     // copy data to the device
    queue.enqueue_write_buffer(gpu_in,0,size_buffers,(void*) gray_image->imageData);
    printf("running with window size %d \n",window_size);
    for (int implementation=1;implementation<=4;implementation++){;
    medianFilter2D_wrapper(queue,foo_program,gpu_in,gpu_out,gpu_histogram,heightImage,widthImage,implementation);
    }

	    // copy data back to the host
// transfer results back to the host array 'c'
    queue.enqueue_read_buffer(gpu_out, 0, size_buffers, output_image->imageData);

	
	cvSaveImage("output.jpg",output_image);
	



}