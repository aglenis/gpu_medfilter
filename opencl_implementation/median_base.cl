#include <stdio.h>

__kernel
void MedianFilter2D( __global unsigned char *input,__global unsigned char* output,int widthImage, int heightImage,unsigned int WINDOW_SIZE)
{
    int filter_offset=WINDOW_SIZE/2;
//y and x are oposite the cuda programming model
    unsigned int y = get_global_id(0);
    unsigned int x = get_global_id(1);;
    if(y>heightImage || x>widthImage)
        return;

    T window[WINDOW_SIZE*WINDOW_SIZE];
    for (int counter=0; counter<WINDOW_SIZE*WINDOW_SIZE; counter++)
    {
        window[counter]=0;
    }
    int count=0;
    for( int k=y-filter_offset; k<=y+filter_offset; k++)
    {
        for (int l=x-filter_offset; l<=x+filter_offset; l++)
        {
	    if(k>=0 && l>=0 && k<heightImage && l<widthImage)
	      window[count++]=input[(k)*widthImage+(l)];

        }
    }
    insertionSort(window,WINDOW_SIZE*WINDOW_SIZE);

    output[y*widthImage + x]=window[WINDOW_SIZE*WINDOW_SIZE/2];

}

__kernel
void MedianFilter2D_histogram( __global unsigned char *input,__global unsigned char* output,int widthImage, int heightImage,unsigned int WINDOW_SIZE)
{
    int filter_offset=WINDOW_SIZE/2;
//y and x are oposite the cuda programming model
    unsigned int y = get_global_id(0);
    unsigned int x = get_global_id(1);;
    if(y>heightImage || x>widthImage)
        return;

    unsigned char window[WINDOW_SIZE*WINDOW_SIZE];
    for (int counter=0; counter<WINDOW_SIZE*WINDOW_SIZE; counter++)
    {
        window[counter]=0;
    }
    int count=0;
    for( int k=max(y-filter_offset,0); k<=min(y+filter_offset,heightImage-1); k++)
    {
        for (int l=max(x-filter_offset,0); l<=min(x+filter_offset,widthImage-1); l++)
        {

            window[count++]=input[(k)*widthImage+(l)];

        }
    }
    unsigned int histogram_values[256];
    for( int i=0; i<256; i++)
    {
        histogram_values[i]=0;
    }
    for(int i=0; i<WINDOW_SIZE*WINDOW_SIZE; i++)
    {
        histogram_values[window[i]]++;
    }
    unsigned int target_value=(WINDOW_SIZE*WINDOW_SIZE-1)/2;
    unsigned int curr_sum=0;
    unsigned int curr_count=0;
    for( curr_count=0; curr_count<256; curr_count++)
    {
        curr_sum+=histogram_values[curr_count];
        if(curr_sum>target_value)
        {
            break;
        }
    }

    output[y*widthImage + x]=curr_count;

}

__kernel
void MedianFilter2D_partial (__global unsigned char *input,__global unsigned char* output,int widthImage, int heightImage,unsigned int WINDOW_SIZE)
{
    int filter_offset=WINDOW_SIZE/2;
//y and x are oposite the cuda programming model
    unsigned int y = get_global_id(0);
    unsigned int x = get_global_id(1);
    if(y>heightImage || x>widthImage)
        return;

    unsigned char window[WINDOW_SIZE*WINDOW_SIZE];
    for (int counter=0; counter<WINDOW_SIZE*WINDOW_SIZE; counter++)
    {
        window[counter]=0;
    }
    int count=0;
    for( int k=max(y-filter_offset,0); k<=min(y+filter_offset,heightImage-1); k++)
    {
        for (int l=max(x-filter_offset,0); l<=min(x+filter_offset,widthImage-1); l++)
        {

            window[count++]=input[(k)*widthImage+(l)];

        }
    }
    partialSelection(window,WINDOW_SIZE);

    output[y*widthImage + x]=window[WINDOW_SIZE*WINDOW_SIZE/2];

}

__kernel
void MedianFilter2D_forgetful(__global unsigned char *input,__global unsigned char* output,int widthImage, int heightImage,unsigned int WINDOW_SIZE)
{
    int filter_offset=WINDOW_SIZE/2;
//y and x are oposite the cuda programming model
    unsigned int y = get_global_id(0);
    unsigned int x = get_global_id(1);
    if(y>heightImage || x>widthImage)
        return;

   unsigned char window[WINDOW_SIZE*WINDOW_SIZE];
    for (int counter=0; counter<WINDOW_SIZE*WINDOW_SIZE; counter++)
    {
        window[counter]=0;
    }
    int count=0;
    for( int k=max(y-filter_offset,0); k<=min(y+filter_offset,heightImage-1); k++)
    {
        for (int l=max(x-filter_offset,0); l<=min(x+filter_offset,widthImage-1); l++)
        {

            window[count++]=input[(k)*widthImage+(l)];

        }
    }
    forgetfulSelection(window,WINDOW_SIZE);

    output[y*widthImage + x]=window[WINDOW_SIZE*WINDOW_SIZE/2];

}

 void insertionSort(__global unsigned char window[],int size)
{
    int i , j;
    unsigned char temp;
    for(i = 0; i < size; i++) {
        temp = window[i];
        for(j = i-1; j >= 0 && temp < window[j]; j--) {
            window[j+1] = window[j];
        }
        window[j+1] = temp;
    }
}

 void partialSelection(__global unsigned char * window,int size)
{
    // Order elements (only half of them)WINDOW_SIZE
    //TODO this works with a odd window size to avoid a ceil function
    for (unsigned int j=0; j<(size*size+1)/2; ++j)
    {
        // Find position of minimum element
        int min_index=j;
        for (unsigned int l=j+1; l<size*size; ++l)
            if (window[l] < window[min_index])
                min_index=l;

        // Put found minimum element in its place
        const unsigned char temp=window[j];
        window[j]=window[min_index];
        window[min_index]=temp;
    }
}

 void swap_elements(__global unsigned char* array,int pos1,int pos2)
{
    unsigned char temp=array[pos1];
    array[pos1]=array[pos2];
    array[pos2]=temp;
}

 void extrema_identification(__global unsigned char * window,int start_offset,int size)
{

    //identify the minimum and maximum elements in the array
    unsigned int min_index,max_index;
    min_index=max_index=start_offset;
    unsigned char max_value=window[start_offset];
    unsigned char min_value=window[start_offset];
    for( int i=start_offset+1; i<start_offset+size; i++)
    {
        if(window[i]<min_value)
        {
            min_index=i;
            min_value=window[i];
        }
        if(window[i]>max_value)
        {
            max_index=i;
            max_value=window[i];
        }

    }
    swap_elements(window,min_index,start_offset);
    swap_elements(window,max_index,size-1+start_offset);
}

void extrema_identification2(__global unsigned char * window,int start_offset,int size)
{

    //identify the minimum and maximum elements in the array
    unsigned int min_index,max_index;
    min_index=max_index=start_offset;
   
    unsigned char min_value=window[start_offset];
    for( int i=start_offset+1; i<start_offset+size; i++)
    {
        if(window[i]<min_value)
        {
            min_index=i;
            min_value=window[i];
        }


    }

   
    swap_elements(window,min_index,start_offset);
    

     unsigned char max_value=window[start_offset];
   for( int i=start_offset+1; i<start_offset+size; i++)
    {
        if(window[i]>max_value)
        {
            max_index=i;
            max_value=window[i];
        }
    }
    swap_elements(window,max_index,size-1+start_offset);

}

 void forgetfulSelection(__global unsigned char * window,int size)
{
    int Rn=ceil((float)(size*size)/2)+1;
    extrema_identification2(window,0,Rn);
    int stop_nr=size*size-Rn;
    for (int step=1; step<=(stop_nr); step++)
    {
        window[Rn-1]=window[Rn+step-1];
        extrema_identification2(window,step,Rn-step);
    }
}



void initialize_histogram(__global unsigned int *histogram_array,__global unsigned char *image,int targetHeight,int targetWidth,int widthImage)
{
  int counter=0;
  for( int i=0;i<targetHeight;i++)
  {
    for(int j=0;j<targetWidth;j++)
    {
      histogram_array[image[i*widthImage+j]]++;
    }
  }
}

void remove_element_from_histogram(__global unsigned int * histogram_array,__global unsigned char * array,int heightPos,int widthPos,int heightImage,int widthImage)
{
  histogram_array[array[heightPos*widthImage+widthPos]]--;
}


void add_element_to_histogram(__global unsigned int * histogram_array,__global unsigned char * array,int heightPos,int widthPos,int heightImage,int widthImage)
{
  histogram_array[array[heightPos*widthImage+widthPos]]++;
}

unsigned char compute_mean_from_histogram(__global unsigned int * histogram_array,int window_size)
{
  unsigned int target_value=(window_size*window_size-1)/2;
        unsigned int curr_sum=0;
	int i=0;
		for( i=0;i<256;i++)
	{
	  curr_sum+=histogram_array[i];
	  if(curr_sum>target_value)
	  {break;}
	}
	return (unsigned char)i;
}

__kernel void MedianFilter2D_histogram_fast(__global unsigned char * array,__global unsigned char * out_array,int heightImage,int widthImage)
{
  
//   unsigned int target_value=(WINDOW_SIZE*WINDOW_SIZE-1)/2;
  int r=WINDOW_SIZE/2;
//   printf("r is %d \n",r);
  unsigned int histogram_array[256];
  
    unsigned int x = get_global_id(0);
//     unsigned int j = get_global_id(1);

    for(int i=r;i<(heightImage-r);i++)
  {
      for(int m=0;m<256;m++){histogram_array[m]=0;}
  initialize_histogram(histogram_array,array+(i-r)*widthImage,WINDOW_SIZE,WINDOW_SIZE,widthImage);
  
    for(int j=r;j<(widthImage-r-1);j++)
    {
      T mean_value=compute_mean_from_histogram(histogram_array,WINDOW_SIZE);
      out_array[i*widthImage+j]=mean_value;
                  for(int k=-r;k<=r;k++)
      {
	
//   	if((j+r)!=(widthImage-r-1))
//   	{
	remove_element_from_histogram(histogram_array,array,i+k,j-1,heightImage,widthImage);
//   	printf("To compute element %d %d i am removing element %d %d \n",i,j,i+k,j-1);
 	add_element_to_histogram(histogram_array,array,i+k,j+2,heightImage,widthImage);
//  	printf("To compute element %d %d i am adding element %d %d \n",i,j,i+k,j+2);
//   	}
      }
      
//        unsigned char mean_value=compute_mean_from_histogram(histogram_array,window_size);
//       out_array[i*widthImage+j]=mean_value;

    }
  }
}

__kernel
void histogram2d(__global unsigned char * array,__global unsigned char * out_array,__global unsigned int * histogram_array,int heightImage,int widthImage,int window_size)
{
  
//   unsigned int target_value=(WINDOW_SIZE*WINDOW_SIZE-1)/2;
  int r=window_size/2;
//   printf("r is %d \n",r);
//   unsigned int histogram_array[256];
unsigned int i = get_global_id(0);
  if(i>=r && i<(heightImage-r))
  {
       for(int m=0;m<256;m++){histogram_array[256*i+m]=0;}
  initialize_histogram(&histogram_array[256*i],array+(i-r)*widthImage,window_size,window_size,widthImage);
  
    for(int j=r;j<(widthImage-r-1);j++)
    {
      unsigned char mean_value=compute_mean_from_histogram(&histogram_array[256*i],window_size);
      out_array[i*widthImage+j]=mean_value;
                  for(int k=-r;k<=r;k++)
      {
	
//   	if((j+r)!=(widthImage-r-1))
//   	{
	remove_element_from_histogram(&histogram_array[256*i],array,i+k,j-1,heightImage,widthImage);
//   	printf("To compute element %d %d i am removing element %d %d \n",i,j,i+k,j-1);
 	add_element_to_histogram(&histogram_array[256*i],array,i+k,j+2,heightImage,widthImage);
//  	printf("To compute element %d %d i am adding element %d %d \n",i,j,i+k,j+2);
//   	}
      }
      
//        unsigned char mean_value=compute_mean_from_histogram(histogram_array,window_size);
//       out_array[i*widthImage+j]=mean_value;

    }
  }
}

void call_median_opencl()
{
  switch(implementation)
  {
    case 1:
      
      break;
    
  }
}