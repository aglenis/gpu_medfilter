#include <stdio.h>

template <typename T,unsigned int WINDOW_SIZE>
__global__
void MedianFilter2D( T *input,T* output,int widthImage, int heightImage)
{
    int filter_offset=WINDOW_SIZE/2;
//y and x are oposite the cuda programming model
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    if(y>heightImage || x>widthImage)
        return;

    T window[WINDOW_SIZE*WINDOW_SIZE];
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
    insertionSort(window,WINDOW_SIZE*WINDOW_SIZE);

    output[y*widthImage + x]=window[WINDOW_SIZE*WINDOW_SIZE/2];

}

template <typename T,unsigned int WINDOW_SIZE>
__global__
void MedianFilter2D_histogram( T *input,T* output,int widthImage, int heightImage)
{
    int filter_offset=WINDOW_SIZE/2;
//y and x are oposite the cuda programming model
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    if(y>heightImage || x>widthImage)
        return;

    T window[WINDOW_SIZE*WINDOW_SIZE];
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

template <typename T,unsigned int WINDOW_SIZE>
__global__
void MedianFilter2D_partial( T *input,T* output,int widthImage, int heightImage)
{
    int filter_offset=WINDOW_SIZE/2;
//y and x are oposite the cuda programming model
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    if(y>heightImage || x>widthImage)
        return;

    T window[WINDOW_SIZE*WINDOW_SIZE];
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

template <typename T,unsigned int WINDOW_SIZE>
__global__
void MedianFilter2D_forgetful( T *input,T* output,int widthImage, int heightImage)
{
    int filter_offset=WINDOW_SIZE/2;
//y and x are oposite the cuda programming model
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    if(y>heightImage || x>widthImage)
        return;

    T window[WINDOW_SIZE*WINDOW_SIZE];
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

__device__ void insertionSort(unsigned char window[],int size)
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

__device__ void partialSelection(unsigned char * window,int size)
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

__device__ void swap_elements(unsigned char* array,int pos1,int pos2)
{
    unsigned char temp=array[pos1];
    array[pos1]=array[pos2];
    array[pos2]=temp;
}

__device__ void extrema_identification(unsigned char * window,int start_offset,int size)
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

__device__ void forgetfulSelection(unsigned char * window,int size)
{
    int Rn=ceilf(size*size/2)+1;
    extrema_identification(window,0,Rn+1);

    int stop_nr=size*size-Rn-1;
    for (int step=0; step<(stop_nr); step++)
    {
        window[Rn]=window[Rn+step+1];
        extrema_identification(window,1+step,Rn-step);
    }
}

template <typename T>
void callMedianFilter(T *input,T* output,int widthImage, int heightImage,int window_size, int threadsX, int threadsY,int implentation)
{
    dim3 block(threadsX, threadsY, 1);
    dim3 grid((int)ceil((float)heightImage / block.x),(int) (ceil((float)widthImage / block.y)), 1);
    printf("original image is size %d %d gridx is %d grid y is %d \n",heightImage,widthImage,grid.x,grid.y);
    switch(window_size)
    {
    case 3:
        switch(implentation) {
        case 1:
            MedianFilter2D<T,3><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 2:
            MedianFilter2D_histogram<T,3><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 3:
            MedianFilter2D_partial<T,3><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 4:
            MedianFilter2D_forgetful<T,3><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        }
        break;
    case 5:
        switch(implentation) {
        case 1:
            MedianFilter2D<T,5><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 2:
            MedianFilter2D_histogram<T,5><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 3:
            MedianFilter2D_partial<T,5><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 4:
            MedianFilter2D_forgetful<T,5><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        }
        break;
    case 7:
        switch(implentation) {
        case 1:
            MedianFilter2D<T,7><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 2:
            MedianFilter2D_histogram<T,7><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 3:
            MedianFilter2D_partial<T,7><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 4:
            MedianFilter2D_forgetful<T,7><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        }
        break;
    case 9:
        switch(implentation) {
        case 1:
            MedianFilter2D<T,9><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 2:
            MedianFilter2D_histogram<T,9><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 3:
            MedianFilter2D_partial<T,9><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        case 4:
            MedianFilter2D_forgetful<T,9><<<grid, block>>>(input,output,widthImage,heightImage);
            break;
        }
        break;
    default:
        printf("That window size has not been implemented yet \n");
        //cudaThreadSynchronize();
    }
}

void MedianFilterUcharCUDA(unsigned char *input,unsigned char* output,int widthImage, int heightImage,int window_size, int threadsX, int threadsY,int implentation)
{
    callMedianFilter<unsigned char>(input,output,widthImage,heightImage,window_size, threadsX, threadsY,implentation);
}