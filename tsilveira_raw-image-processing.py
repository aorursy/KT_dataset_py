## Mathmatics libraries
import numpy as np
import math

## Image Processing libraries
import skimage
from skimage import exposure

import scipy.misc as misc

import rawpy
import imageio

## Visual and plotting libraries
import matplotlib.pyplot as plt
## Reading a RAW file:
rawImg = rawpy.imread('../input/IMG_0978.CR2')
#rgbImg = rawImg.postprocess()
rgbImg = rawImg.raw_image_visible
type(rgbImg)
def basic_showImg(img, size=4):
    '''Shows an image in a numpy.array type. Syntax:
        basic_showImg(img, size=4), where
            img = image numpy.array;
            size = the size to show the image. Its value is 4 by default.
    '''
    plt.figure(figsize=(size,size))
    plt.imshow(img)
    plt.show()
basic_showImg(rgbImg,8)
# Gamma adjustment
gamma_corrected = exposure.adjust_gamma(rgbImg, gamma=0.5, gain=1)
basic_showImg(gamma_corrected,8)
# Histogram equalization
hist_equalized = exposure.equalize_hist(rgbImg)
basic_showImg(hist_equalized,8)
def basic_writeImg(directory, filename, img):
    imageio.imwrite(directory+filename, img)
basic_writeImg('','edited_img.png', hist_equalized)
