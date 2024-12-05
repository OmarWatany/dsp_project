Input signal --> ecg400 & Filter Specifications
Output should match --> ecg_band_pass_filtered
filter specification file is used to determine the filter type, window type, number of coefficients, etc ....

Here, the output is the filtered signal which is y(n)=x(n)*h(n),where h(n) = hd(n) x w(n) and here you will return
the filtered signal after applying convolution and your output should match ecg_band_pass_filtered file output