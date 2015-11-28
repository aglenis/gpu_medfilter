#include <algorithm>    // std::sort
#include "../ccdf_test/ccdf_test.cpp"

template <typename T,unsigned int WINDOW_SIZE>
void MedianFilter2D( T *input,T* output,int widthImage, int heightImage)
{
int filter_offset=WINDOW_SIZE/2;
 for (int y=0;y<(heightImage);y++)
 {
   for (int x=0;x<(widthImage);x++)
   {
    
     T window[WINDOW_SIZE*WINDOW_SIZE];
     for (int counter=0;counter<WINDOW_SIZE*WINDOW_SIZE;counter++)
     {
       window[counter]=0;
     }
 
     for( int k=0;k<WINDOW_SIZE;k++)
     {
       for (int l=0;l<WINDOW_SIZE;l++)
       {
	 if((y+k-filter_offset)<0 ||(y+k-filter_offset)>heightImage ||(x+l-filter_offset)<0 ||(x+l-filter_offset)>widthImage)
	 {
	   continue;
	}
	else
	{
	 window[k*WINDOW_SIZE+l]=input[(y+k-filter_offset)*widthImage+(x+l-filter_offset)];
	}

	   }
       }
       std::sort(window,window+WINDOW_SIZE*WINDOW_SIZE);

    output[y*widthImage + x]=window[WINDOW_SIZE*WINDOW_SIZE/2];
   }
 }
}

template <typename T,unsigned int WINDOW_SIZE>
void MedianFilter2D2( T *input,T* output,int widthImage, int heightImage)
{
int filter_offset=WINDOW_SIZE/2;
 for (int y=0;y<(heightImage);y++)
 {
   for (int x=0;x<(widthImage);x++)
   {
    
     T window[WINDOW_SIZE*WINDOW_SIZE];
     for (int counter=0;counter<WINDOW_SIZE*WINDOW_SIZE;counter++)
     {
       window[counter]=0;
     }
 int count=0;
      for( int k=std::max(y-filter_offset,0);k<=std::min(y+filter_offset,heightImage-1);k++)
     {
       for (int l=std::max(x-filter_offset,0);l<=std::min(x+filter_offset,widthImage-1);l++)
       {

	 window[count++]=input[(k)*widthImage+(l)];

	   }
       }
       insertionSort(window,WINDOW_SIZE*WINDOW_SIZE);

    output[y*widthImage + x]=window[WINDOW_SIZE*WINDOW_SIZE/2];
   }
 }
}

template <typename T,unsigned int WINDOW_SIZE>
void MedianFilter2D3( T *input,T* output,int widthImage, int heightImage)
{
int filter_offset=WINDOW_SIZE/2;
 for (int y=0;y<(heightImage);y++)
 {
   for (int x=0;x<(widthImage);x++)
   {
    
     T window[WINDOW_SIZE*WINDOW_SIZE];
     for (int counter=0;counter<WINDOW_SIZE*WINDOW_SIZE;counter++)
     {
       window[counter]=0;
     }
 int count=0;
      for( int k=std::max(y-filter_offset,0);k<=std::min(y+filter_offset,heightImage-1);k++)
     {
       for (int l=std::max(x-filter_offset,0);l<=std::min(x+filter_offset,widthImage-1);l++)
       {

	 window[count++]=input[(k)*widthImage+(l)];

	   }
       }
       int Rn=ceil(WINDOW_SIZE*WINDOW_SIZE/2)+1;
       insertionSort(window,Rn);
       T temp;
       for (int step=0;step<WINDOW_SIZE*WINDOW_SIZE-Rn;step++)
       {
	 window[Rn-1]=window[Rn+step];
	 int min_pos=Rn-1;;
	 for(int j=Rn-2;step+1;j--)
	 {
	   if(window[Rn-1]<window[j])
	   {
	     min_pos=j;
	  }
	  temp=window[min_pos];
	  window[min_pos]=window[Rn-1];
	  window[Rn-1]=temp;
	}
      }

    output[y*widthImage + x]=window[WINDOW_SIZE*WINDOW_SIZE/2];
   }
 }
}


template <typename T,unsigned int WINDOW_SIZE>
void MedianFilter2D4( T *input,T* output,int widthImage, int heightImage)
{
int filter_offset=WINDOW_SIZE/2;
 for (int y=0;y<(heightImage);y++)
 {
   for (int x=0;x<(widthImage);x++)
   {
    
     T window[WINDOW_SIZE*WINDOW_SIZE];
     for (int counter=0;counter<WINDOW_SIZE*WINDOW_SIZE;counter++)
     {
       window[counter]=0;
     }
 int count=0;
      for( int k=std::max(y-filter_offset,0);k<=std::min(y+filter_offset,heightImage-1);k++)
     {
       for (int l=std::max(x-filter_offset,0);l<=std::min(x+filter_offset,widthImage-1);l++)
       {

	 window[count++]=input[(k)*widthImage+(l)];  
	 
      }
       }
       unsigned int histogram_values[256];
	for( int i=0;i<256;i++)
	{
	  histogram_values[i]=0;
	}
	for(int i=0;i<WINDOW_SIZE*WINDOW_SIZE;i++)
	{
	  histogram_values[window[i]]++;
	}
      
      unsigned int target_value=(WINDOW_SIZE*WINDOW_SIZE-1)/2;
	unsigned int curr_sum=0;
	unsigned int i=0;
		for( i=0;i<256;i++)
	{
	  curr_sum+=histogram_values[i];
	  if(curr_sum>target_value)
	  {break;}
	}

    output[y*widthImage + x]=i;
   }
 }
}


template <typename T,unsigned int WINDOW_SIZE>
void MedianFilter2D5( T *input,T* output,int widthImage, int heightImage)
{
int filter_offset=WINDOW_SIZE/2;
unsigned int target_value=(WINDOW_SIZE*WINDOW_SIZE-1)/2;
       unsigned int histogram_values[256];
	for( int i=0;i<256;i++)
	{
	  histogram_values[i]=0;
	}
	
	//here we initialize the histogram for the first time
	int y_start=0;
	int y_end=WINDOW_SIZE;
	int x_start=0;
	int x_end=WINDOW_SIZE;
	
	for( int init_y=y_start;init_y<y_end;init_y++)
	{
	  for (int init_x=x_start;init_x<x_end;init_x++)
	  {
	    histogram_values[input[(init_y)*widthImage*(init_x)]]++;
	  }
	}
 for (int y=0+filter_offset;y<=(heightImage-filter_offset);y++)
 {
   for (int x=0+filter_offset;x<=(widthImage-filter_offset);x++)
   {
     for( int k=-filter_offset;k<=filter_offset;k++)
     {
       histogram_values[input[(y+k)*widthImage+(x-filter_offset-1)]]--;
       histogram_values[input[(y+k)*widthImage+(x+filter_offset)]]++;
    }
    
     
//      T window[WINDOW_SIZE*WINDOW_SIZE];
//      for (int counter=0;counter<WINDOW_SIZE*WINDOW_SIZE;counter++)
//      {
//        window[counter]=0;
//      }
//  int count=0;
//       for( int k=std::max(y-filter_offset,0);k<=std::min(y+filter_offset,heightImage-1);k++)
//      {
//        for (int l=std::max(x-filter_offset,0);l<=std::min(x+filter_offset,widthImage-1);l++)
//        {
// 
// 	 window[count++]=input[(k)*widthImage+(l)];  
// 	 
//       }
//        }
// 
// 	for(int i=0;i<WINDOW_SIZE*WINDOW_SIZE;i++)
// 	{
// 	  histogram_values[window[i]]++;
// 	}
      
      
	unsigned int curr_sum=0;
	unsigned int i=0;
		for( i=0;i<256;i++)
	{
	  curr_sum+=histogram_values[i];
	  if(curr_sum>target_value)
	  {break;}
	}

    output[y*widthImage + x]=0;
   }
 }
}

void initialize_histogram(unsigned int *histogram_array,unsigned char *image,int targetHeight,int targetWidth,int widthImage)
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

void initialize_ccdf(unsigned int *target_input,unsigned char *image,int targetHeight,int targetWidth,int widthImage,int start_value_range,int max_value_range)
{
  perform_ccdf2d(image,target_input,0,targetHeight,0,targetWidth,start_value_range,max_value_range);
}

void remove_element_from_histogram(unsigned int * histogram_array,unsigned char * array,int heightPos,int widthPos,int heightImage,int widthImage)
{
  histogram_array[array[heightPos*widthImage+widthPos]]--;
}


void add_element_to_histogram(unsigned int * histogram_array,unsigned char * array,int heightPos,int widthPos,int heightImage,int widthImage)
{
  histogram_array[array[heightPos*widthImage+widthPos]]++;
}

unsigned char compute_mean_from_histogram(unsigned int * histogram_array,int window_size)
{
  unsigned int target_value=(window_size*window_size-1)/2;
        unsigned int curr_sum=0;
	unsigned char i=0;
		for( i=0;i<256;i++)
	{
	  curr_sum+=histogram_array[i];
	  if(curr_sum>target_value)
	  {break;}
	}
	return i;
}

void insertionSort(unsigned char window[],int size)
{
    int i , j;
    unsigned char temp;
    for(i = 0; i < size; i++){
        temp = window[i];
        for(j = i-1; j >= 0 && temp < window[j]; j--){
            window[j+1] = window[j];
        }
        window[j+1] = temp;
    }
}

void histogram2d(unsigned char * array,unsigned char * out_array,int heightImage,int widthImage,int window_size)
{
  
//   unsigned int target_value=(WINDOW_SIZE*WINDOW_SIZE-1)/2;
  int r=window_size/2;
//   printf("r is %d \n",r);
  unsigned int * histogram_array=(unsigned int *)malloc(sizeof(unsigned int)*256);

  for(int i=r;i<(heightImage-r);i++)
  {
      for(int m=0;m<256;m++){histogram_array[m]=0;}
  initialize_histogram(histogram_array,array+(i-r)*widthImage,window_size,window_size,widthImage);
  
    for(int j=r;j<(widthImage-r-1);j++)
    {
      unsigned char mean_value=compute_mean_from_histogram(histogram_array,window_size);
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

void histogram_ccdf(unsigned char * array,unsigned char * out_array,int heightImage,int widthImage,int window_size)
{
  
//   unsigned int target_value=(WINDOW_SIZE*WINDOW_SIZE-1)/2;
  int r=window_size/2;
//   printf("r is %d \n",r);
  int start_range=0;
  int end_range=255;
  
  int new_end_value_range=(window_size*window_size)/2-1;
  
  unsigned int * input_array=(unsigned int *)malloc(sizeof(unsigned int)*(end_range-start_range));
  unsigned int * output_array=(unsigned int *)malloc(sizeof(unsigned int)*(new_end_value_range+1));
  unsigned int * ccdf_array=(unsigned int * )malloc(sizeof(unsigned int)*window_size);
  unsigned int * temp_output_array=(unsigned int *)malloc(sizeof(unsigned int)*(new_end_value_range+1));
  unsigned char * curr_window=(unsigned char *)malloc(sizeof(unsigned char)*window_size*window_size);
  
  

  for(int i=r;i<(heightImage-r);i++)
  {

//   initialize_ccdf(input_array,array+(i-r)*widthImage,window_size,window_size,widthImage,start_range,end_range);
  int count;
    for(int j=r;j<(widthImage-r);j++)
    {
//       unsigned char mean_value=compute_mean_from_histogram(histogram_array,window_size);
//       unsigned char mean_value=(input_array,output_array,start_range,end_range,0,new_end_value_range);
//       unsigned char mean_value=(unsigned char)input_array[(window_size*window_size)/2];
//       out_array[i*widthImage+j]=mean_value;
//       printf("mean value is %u \n",mean_value);
      count=0;
      int window_count=0;
      
                 for(int k=-r;k<=r;k++)
      {
	    for(int l=-r;l<=r;l++)
      {
	curr_window[window_count++]=array[(i+k)*widthImage+(j+l)];
      }
      }
// unsigned char mean_value=median_from_ccdf(curr_window,output_array,temp_output_array,start_range,end_range,0,window_size*window_size);
insertionSort(curr_window,window_size*window_size);
unsigned char mean_value=curr_window[window_size*window_size/2];
printf("mean value is %u \n",mean_value);

out_array[i*widthImage+j]=mean_value;
      /*
      //perform ccdf of the values to remove
                  for(int k=-r;k<=r;k++)
      {
	ccdf_array[count++]=array[(i+k)*widthImage+(j-1)];
      }
      printf("printing values to be removed \n");
      for(int count1=0;count1<window_size;count1++)
      {
	printf("%d ",ccdf_array[count1]);
      }
      printf("\n");
//
      perform_ccdf(ccdf_array,temp_output_array,0,new_end_value_range,start_range,end_range);
      printf("printing the results after ccdf \n");
      for( int count1=0;count1<new_end_value_range+1;count1++)
      {
	printf("%u ",temp_output_array[count1]);
      }
      printf("\n");
      for( int count1=0;count1<=new_end_value_range;count1++)
      {
	output_array[count1]-=temp_output_array[count1];
      }

      count=0;
      //perform ccdf of the values to add
                for(int k=-r;k<=r;k++)
      {
	ccdf_array[count++]=array[(i+k)*widthImage+(j+2)];
      }
      
      printf("printing values to be added \n");
      for(int count1=0;count1<window_size;count1++)
      {
	printf("%d ",ccdf_array[count1]);
      }
      printf("\n");
      
      perform_ccdf(ccdf_array,temp_output_array,0,new_end_value_range,start_range,end_range);
      printf("printing the results of add ccdf \n");
          for( int count1=0;count1<new_end_value_range+1;count1++)
      {
	printf("%u ",temp_output_array[count1]);
      }
      printf("\n");
      for( int i=0;i<=new_end_value_range;i++)
      {
	output_array[i]+=temp_output_array[i];
      }
      */
    }
  }
}

template <typename T,unsigned int WINDOW_SIZE>
void MedianFilter2D6( T *input,T* output,int widthImage, int heightImage)
{
histogram2d(input,output,heightImage,widthImage,WINDOW_SIZE);
}


