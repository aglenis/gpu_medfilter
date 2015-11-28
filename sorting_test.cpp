#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "sorting_implementations.cpp"
#include <time.h>
#include "histogram2do1.cpp"


int main()
{
  int WINDOW_SIZE=3;
  srand(time(0));
  unsigned char original_window[WINDOW_SIZE*WINDOW_SIZE];
  for (int i=0;i<WINDOW_SIZE*WINDOW_SIZE;i++)
  {
    original_window[i]=rand();
  }
  unsigned char selection_window[WINDOW_SIZE*WINDOW_SIZE];
  unsigned char insertion_window[WINDOW_SIZE*WINDOW_SIZE];
  unsigned char forgetful_window[WINDOW_SIZE*WINDOW_SIZE];
  unsigned char histogram_window[WINDOW_SIZE*WINDOW_SIZE];
  
    for (int i=0;i<WINDOW_SIZE*WINDOW_SIZE;i++)
  {
    selection_window[i]=original_window[i];
    insertion_window[i]=original_window[i];
    forgetful_window[i]=original_window[i];
    histogram_window[i]=original_window[i];
  }
  
  printf("the array before any operation done to it is: \n");
  print_array(original_window,0,WINDOW_SIZE*WINDOW_SIZE);

  insertionSort(insertion_window,WINDOW_SIZE*WINDOW_SIZE);
  unsigned char median_value1=insertion_window[WINDOW_SIZE*WINDOW_SIZE/2];
  partialSelection(selection_window,WINDOW_SIZE);
  unsigned char median_value2=selection_window[WINDOW_SIZE*WINDOW_SIZE/2];
   forgetfulSelection(forgetful_window,WINDOW_SIZE);
  unsigned char median_value3=forgetful_window[WINDOW_SIZE*WINDOW_SIZE/2];
  
  printf("median_value1 is %d median_value2 is %d median_value3 is %d \n",median_value1,median_value2,median_value3);
  


#if 1
	unsigned int histogram_values[256];
	for( int i=0;i<256;i++)
	{
	  histogram_values[i]=0;
	}
	for(int i=0;i<WINDOW_SIZE*WINDOW_SIZE;i++)
	{
	  histogram_values[histogram_window[i]]++;
	}
		for(int i=0;i<256;i++)
	{
	  printf(" %d ",histogram_values[i]);
	}
	printf("\n");

unsigned char median_value4=compute_mean_from_histogram(histogram_values,WINDOW_SIZE);
 
	      unsigned char median_value5=selection_window[WINDOW_SIZE*WINDOW_SIZE/2];
	      
	      printf("median_value1 is %u median_value4 is %u \n",median_value1,median_value5);
//       printf("median_value1 is %u median_value2 is %u \n",median_value1,median_value2);
#endif    
      
  
}
