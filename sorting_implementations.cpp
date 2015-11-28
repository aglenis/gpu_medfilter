#include <stdio.h>

void print_array(unsigned char * array,int start_index,int size)
{
  for (int counter=start_index;counter<start_index+size;counter++)
  {
    printf("%d ",array[counter]);
  }
  printf("\n");
}

void insertionSort(unsigned char window[],int size)
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

void partialSelection(unsigned char * window,int size)
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

void swap_elements(unsigned char* array,int pos1,int pos2)
{
    unsigned char temp=array[pos1];
    array[pos1]=array[pos2];
    array[pos2]=temp;
}

void extrema_identification(unsigned char * window,int start_offset,int size)
{
printf("the array in the window of interest before extrema_identification is: \n");
print_array(window,start_offset,size);
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
    printf("maximum element is %d with index %d \n ",max_value,max_index);
    printf("minimum element is %d with index %d \n ",min_value,min_index);
   
    swap_elements(window,max_index,size-1+start_offset);
    swap_elements(window,min_index,start_offset);
    
     if(max_index==start_offset || min_index==(size+start_offset-1)){
       printf("probably a problem in this run \n");
	 swap_elements(window,max_index,start_offset);
     }
//      else
//      {
// 	
//     }
    
    printf("the array in the window of interest after extrema_identification is: \n");
    print_array(window,start_offset,size);
}

void extrema_identification2(unsigned char * window,int start_offset,int size)
{
printf("the array in the window of interest before extrema_identification is: \n");
print_array(window,start_offset,size);
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
    printf("minimum element is %d with index %d \n ",min_value,min_index);
   
    swap_elements(window,min_index,start_offset);
    
//      else
//      {
// 	
//     }
     unsigned char max_value=window[start_offset];
   for( int i=start_offset+1; i<start_offset+size; i++)
    {
        if(window[i]>max_value)
        {
            max_index=i;
            max_value=window[i];
        }
    }
    printf("maximum element is %d with index %d \n ",max_value,max_index);
    swap_elements(window,max_index,size-1+start_offset);
    printf("the array in the window of interest after extrema_identification is: \n");
    print_array(window,start_offset,size);
}

void forgetfulSelection(unsigned char * window,int size)
{
  
    int Rn=ceil((float)(size*size)/2)+1;
    printf("Rn is %d \n",Rn);
    extrema_identification2(window,0,Rn);

    int stop_nr=size*size-Rn;
    printf("stop_nr is %d \n",stop_nr);
    for (int step=1; step<=(stop_nr); step++)
    {
        window[Rn-1]=window[Rn+step-1];
        extrema_identification2(window,step,Rn-step);
    }
}
