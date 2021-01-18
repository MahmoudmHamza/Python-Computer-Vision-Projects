# Computer Vision Toolbox

> **NOTE**

This project was part of a classwork during the whole course so Ii'm totally aware that it's neither written in a clean way nor following Python best practices.

## Overview

This toolbox includes image processing filters implemented using python 3.6.

The tool box's UI (User Interface) was implemented using PyQt5 library.

![UI](https://user-images.githubusercontent.com/44654878/56047144-18b2ac00-5d45-11e9-9779-be28a965f72e.PNG)

![UI2](https://user-images.githubusercontent.com/44654878/56047234-4566c380-5d45-11e9-973c-c879bf654562.PNG)

> Spatial domain filtering

The spatial domain filtering technique operates directly on individual pixels as it includes point process dealing with single point, mask process dealing with a window of pixels, global process dealing with the whole image.

- Edge detection filters
  - Prewitt filter
  
  The Prewitt operator is based on convolving the image with integer-valued filter in horizontal and vertical directions
  
  ![Prewitt kernel](https://user-images.githubusercontent.com/44654878/56047562-f66d5e00-5d45-11e9-8c16-f04b39b16a79.PNG)
  
  X Direction
  
  ![Prewitt_Gx](https://user-images.githubusercontent.com/44654878/56066928-955f7d80-5d79-11e9-87e6-82d3a71198e7.PNG)
  
  Y Direction
  
  ![Prewitt_Gy](https://user-images.githubusercontent.com/44654878/56066929-955f7d80-5d79-11e9-81dd-b35c325bc02a.PNG)
  
  - Sobel filter
  
  The Sobel operator is based on convolving the image with integer-valued filter in the horizontal and vertical directions
  
  ![Sobel kernel](https://user-images.githubusercontent.com/44654878/56047564-f705f480-5d45-11e9-826c-dd2911ea1246.PNG)
  
  X Direction
  
  ![Sobel](https://user-images.githubusercontent.com/44654878/56066931-95f81400-5d79-11e9-9a7b-1853ba4d1bcc.PNG)
  
  Y Direction
  
  ![Sobel_Gy](https://user-images.githubusercontent.com/44654878/56066933-95f81400-5d79-11e9-90c1-a9692c66d2e1.PNG)
  
  - Laplacian
  
  edges are identiﬁed with zero-crossings of second-order derivatives using the following kernels
  
  ![Laplacian](https://user-images.githubusercontent.com/44654878/56066924-94c6e700-5d79-11e9-9bee-1a81bea1d96c.PNG)
  
  - Laplacian of Gaussian
  
  Applying the Laplacian for a Gauss-ﬁltered image through convolution
  
  ![LOG](https://user-images.githubusercontent.com/44654878/56066926-94c6e700-5d79-11e9-882c-bbbdf9abe412.PNG)
  
  - Diﬀerence of Gaussians
  
  The DoG represents an approximation of a convolution of I with the partial derivative
  
  ![DOG](https://user-images.githubusercontent.com/44654878/56067462-824dad00-5d7b-11e9-83b6-6722c49cf998.jpeg)
  
- Smoothing filters
  - Box filter
  
  this filter removes image outliers but affects the contrast of it
  
  ![Box](https://user-images.githubusercontent.com/44654878/56066935-9690aa80-5d79-11e9-8225-51fae184107d.PNG)
  
  - Median filter
  
  removing the image outliers as above with a pretty small effect on image contrast
  
  ![Median](https://user-images.githubusercontent.com/44654878/56066927-955f7d80-5d79-11e9-8590-ad80f8861e48.PNG)
  
  - Gaussian
  
  The Gaussian smoothing operator is a 2-D convolution operator that is used to blur images and remove detail and noise as following
  
  ![Gaussian kernel](https://user-images.githubusercontent.com/44654878/56047556-f5d4c780-5d45-11e9-8df5-8f1159435f74.PNG)
  
  ![Gaussian](https://user-images.githubusercontent.com/44654878/56066936-9690aa80-5d79-11e9-9179-9847d27ba1fb.PNG)
  
- Sharpening

  Sharpening is used to produce an enhanced image by increasing the contrast without adding too much noise
  
  ![Sharpening](https://user-images.githubusercontent.com/44654878/56066930-955f7d80-5d79-11e9-9482-d323f8d00e22.PNG)
  
> Frequency domain filtering

The frequency domain filtering technique changes the frequency components of an image as it include global process.

At the beginning, we have to apply fourier transform into the input image so as to move it from the spatial domain to frequency domain, after then we can use frequency domain filters.

![FFT](https://user-images.githubusercontent.com/44654878/56049399-abede080-5d49-11e9-90b7-431ca13f852c.jpeg)

- High Pass Filter

This filter is used for edge detection in the frequency domain

![HPF](https://user-images.githubusercontent.com/44654878/56049397-abede080-5d49-11e9-958f-8a51d5838fcd.jpeg)

- Low Pass Filter

This filter is used for Blurring and smoothing in the frequency domain

![LPF](https://user-images.githubusercontent.com/44654878/56049398-abede080-5d49-11e9-9785-bc58db9fbac6.jpeg)
 
> Hough Transform

The Hough transform is a technique which can be used to isolate features of a particular shape within an image.

![Hough circle](https://user-images.githubusercontent.com/44654878/56067218-aceb3600-5d7a-11e9-97a2-b544a117288b.jpeg)

> Histogram matching

![Histogram](https://user-images.githubusercontent.com/44654878/56049396-abede080-5d49-11e9-85a2-9c86a043e1e6.jpeg)

> Corner detection

> Snake algorithm

> Image segmentation and clustering
- Region growing
- Kmeans
- Mean Shift

### Acknowledgment

A huge shoutout for @AbdelrahmanARamzy who helped us tackling some issues in UI part.

### References

- This [link](https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123) helped in the implementation of canny edge detector.
- This [link](https://www.codingame.com/playgrounds/38470/how-to-detect-circles-in-images) helped in the implementation of hough circle.
- We used this [link](https://www.youtube.com/watch?v=YxZUnJ_Ok2w&list=PLh6SAYydrIpctChfPFBlopqw-TGjwWf_8&index=8) in histogram equalization and matching implementation.
