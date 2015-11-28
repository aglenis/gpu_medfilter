#include "histogram2do1.cpp"
#include <cstdlib>
#include <stdio.h>

int main()

{
  int numRows=3;
  int numColumns=4;
  int window_size=3;
  
  unsigned char * array=(unsigned char *)malloc(numRows*numColumns*sizeof(unsigned char));
  
  for (int i=0;i<numRows*numColumns;i++)
  {
    array[i]=rand()%256;
  }
  printf("original array is : \n");
  for (int i=0;i<numRows;i++)
  {
    for(int j=0;j<numColumns;j++)
    {
      printf("%u ",array[i*numColumns+j]);
    }
    printf("\n");
  }
  
  unsigned char * out_array=(unsigned char *)malloc(numRows*numColumns*sizeof(unsigned char));
    for (int i=0;i<numRows*numColumns;i++)
  {
    out_array[i]=0;
  }
 histogram2d(array,out_array,numRows,numColumns,window_size);
 
  printf("histogram array is : \n");
  for (int i=0;i<numRows;i++)
  {
    for(int j=0;j<numColumns;j++)
    {
      printf("%u ",out_array[i*numColumns+j]);
    }
    printf("\n");
  }
}