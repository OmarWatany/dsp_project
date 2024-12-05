Input signal --> Filter Specifications
Output should match --> BSFCoefficients
filter specification file is used to determine the filter type, window type, number of coefficients, etc ....

Here, the output is the finite filter in the time domain which is h(n) = hd(n) x w(n) and here you will not call
the convolution function to apply the filter on the signal, you will only return the filter and it should match 
BSFCoefficients file output.