#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include "../timer.h"
#include "median_base.cpp"


int main(int argc, char** argv)
{

	float fTotalTime = 0;


	const unsigned int kernel_size=3;
// 	IplImage* gray_image = cvLoadImage(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
	int widthImage=5;
	int heightImage=5;
	
	unsigned char * input_image=(unsigned char *)malloc(widthImage*heightImage*sizeof(unsigned char));
	
	for( int i=0;i<heightImage;i++)
	  for( int j=0;j<widthImage;j++)
	    input_image[i*widthImage+j]=i*widthImage+j;
	
	unsigned char * output_image=(unsigned char *)malloc(widthImage*heightImage*sizeof(unsigned char));
	for( int i=0;i<heightImage;i++)
	  for( int j=0;j<widthImage;j++)
	    output_image[i*widthImage+j]=0;

histogram_ccdf(input_image,output_image,heightImage,widthImage,kernel_size);  
printf("printing output array \n");

      for( int i=0;i<heightImage;i++){
	printf("\n");
	  for( int j=0;j<widthImage;j++)
	  {
	    printf("%u ",output_image[i*widthImage+j]);
	  }
      }
      printf("\n");
	

}