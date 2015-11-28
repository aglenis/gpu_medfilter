#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include "../timer.h"
#include "median_base.cpp"


int main(int argc, char** argv)
{

	float fTotalTime = 0;

// 	int TARGET_WIDTH=atoi(argv[2]);
// 	int TARGET_HEIGHT=atoi(argv[3]);
// 	bool visualize_results=atoi(argv[4]);
// 	int gpuNr=atoi(argv[5]);
// 	checkCudaErrors(cudaSetDevice(gpuNr));

	const unsigned int kernel_size=3;
	IplImage* gray_image = cvLoadImage(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
	int widthImage=gray_image->width;
	int heightImage=gray_image->height;
	
	IplImage *output_image = cvCreateImage(cvSize(widthImage,heightImage), IPL_DEPTH_8U, 1);
	for( int i=0;i<heightImage;i++)
	  for( int j=0;j<widthImage;j++)
	    output_image->imageData[i*widthImage+j]=255;
// 	MedianFilter2D6<unsigned char,kernel_size>((unsigned char *)gray_image->imageData,(unsigned char *)output_image->imageData,widthImage,heightImage);
// 	_medianfilter((unsigned char *)gray_image->imageData, (unsigned char *)output_image->imageData, widthImage, heightImage);
histogram_ccdf((unsigned char *)gray_image->imageData,(unsigned char *)output_image->imageData,heightImage,widthImage,kernel_size);  
	
	cvSaveImage("output.jpg",output_image);
	



}