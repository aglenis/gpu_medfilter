 #include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include "../timer.h"
#include "median_cuda.h"
#include "arrayFire_imp.h"


int main(int argc, char** argv)
{

	float fTotalTime = 0;

// 	int TARGET_WIDTH=atoi(argv[2]);
// 	int TARGET_HEIGHT=atoi(argv[3]);
// 	bool visualize_results=atoi(argv[4]);
// 	unsigned int kernel_size=atoi(argv[2]);
 	int gpuNr=atoi(argv[2]);
 	checkCudaErrors(cudaSetDevice(gpuNr));

	
	IplImage* gray_image = cvLoadImage(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
	unsigned char * d_input_image;
	unsigned char * d_output_image;
	int widthImage=gray_image->width;
	int heightImage=gray_image->height;
	
	IplImage *output_image = cvCreateImage(cvSize(widthImage,heightImage), IPL_DEPTH_8U, 1);
	for( int i=0;i<heightImage;i++)
	  for( int j=0;j<widthImage;j++)
	    output_image->imageData[i*widthImage+j]=255;
	
	  	  unsigned int * d_histogram;
	int total_threads=256;	  
	  cudaMalloc(&d_histogram,sizeof(unsigned int)*256*total_threads);
	checkCudaErrors(cudaMalloc(&d_input_image,widthImage*heightImage*sizeof(unsigned char)));  
	checkCudaErrors(cudaMalloc(&d_output_image,widthImage*heightImage*sizeof(unsigned char)));
	checkCudaErrors(cudaMemcpy(d_input_image,gray_image->imageData,widthImage*heightImage*sizeof(unsigned char),cudaMemcpyHostToDevice));
	unsigned int windows_array[4]={3,5,7,9};
	int total_implementations=4;
	double elapsed_time;
	for (int i=1;i<=total_implementations;i++)
	{
	  for( int j=0;j<4;j++)
	  {
	timer my_timer;  
	MedianFilterUcharCUDA(d_input_image,d_output_image,d_histogram,widthImage,heightImage,windows_array[j],16,16,i);
	cudaThreadSynchronize();
	elapsed_time=my_timer.elapsed();
	printf("elapsed_time for implementation %d for window size %d was %f \n",i,windows_array[j],elapsed_time);
	  }
	}
	timer array_timer;
	arrayFireRows(d_input_image,d_output_image,widthImage,heightImage,3,16,16);
		cudaThreadSynchronize();
	elapsed_time=array_timer.elapsed();
	printf("elapsed_time for array fire was %f \n",elapsed_time);
	checkCudaErrors(cudaMemcpy(output_image->imageData,d_output_image,widthImage*heightImage*sizeof(unsigned char),cudaMemcpyDeviceToHost));
// 	_medianfilter((unsigned char *)gray_image->imageData, (unsigned char *)output_image->imageData, widthImage, heightImage);
	
	cvSaveImage("output.jpg",output_image);
	



}