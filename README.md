# gpu_medfilter
Median filter implementation on GPUs and multicores

This project contains CUDA and OpenCL implementations of the various ways one can implement a median filter.

There is also a serial implementation to test for correctness and as a starting point for creating speedup graphs.

The implementation uses OpenCV to open the image.
I have tested it on ubuntu 14.04 with the NVIDIA drivers and CUDA runtime.

I worked through the bibliography and implemented versions based on the following papers:

BIDIMENSIONAL MEDIAN FILTER FOR PARALLEL COMPUTING ARCHITECTURES by R.M. Sanchez et al

Fine-tuned high-speed implementation of a GPU-based median filter by Perrot et al

Another great reference is :
Median Filtering in Constant Time by Perreault et al
