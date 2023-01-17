#import cv2
#import keras
#import sklearn

#import skimage
from skimage import io, data

## I tryied to load the BOVESPA image but it didnt work. It says that it doesnt recognize the file path.

## I used io.imread('../input/imgtesting/BOVESPA-prego-trading-post-inicio-70-1.jpg')

cell_img = data.cell()

io.imshow(cell_img)
astronaut_img = data.astronaut()

io.imshow(astronaut_img)
#Getting image resolution, and number of channels

astronaut_img.shape
#Getting pixel values

import pandas as pd

df = pd.DataFrame(astronaut_img.flatten())

#file_path = 'pixel_values_1.xlsx'

#df.to_excel(file_path, index=False)
astronaut_img.flatten()
df
astronaut_img
from skimage import color

from pylab import *
#Converting from RGB to HSV:

img_hsv = color.rgb2hsv(astronaut_img)



#Converting back from HSV to RGB:

img_hsv_rgb = color.hsv2rgb(img_hsv)
img_hsv.flatten()
img_hsv_rgb.flatten()
#Showing both images:

figure(0) #function figure in module matplotlib.pyplot; Create a new figure.

io.imshow(img_hsv)

figure(1)

io.imshow(img_hsv_rgb)
#Converting from RGB to XYZ

img_xyz = color.rgb2xyz(astronaut_img)

#and back to rgb

img_xyz_rgb = color.xyz2rgb(img_xyz)



figure(0)

io.imshow(img_xyz)

figure(1)

io.imshow(img_xyz_rgb)
#RGB to LAB:

img_lab = color.rgb2lab(astronaut_img)

#and back to rgb:

img_lab_rgb = color.lab2rgb(img_lab)



figure(0)

io.imshow(img_lab)

figure(1)

io.imshow(img_lab_rgb)
#rgb to YUV

img_yuv = color.rgb2yuv(astronaut_img)

#and back to RGB:

img_yuv_rgb = color.yuv2rgb(img_yuv)



figure(0)

io.imshow(img_yuv)

figure(1)

io.imshow(img_yuv_rgb)
#RGB to YIQ:

img_yiq = color.rgb2yiq(astronaut_img)

#back to RGB:

img_yiq_rgb = color.yiq2rgb(img_yiq)



figure(0)

io.imshow(img_yiq)

figure(1)

io.imshow(img_yiq_rgb)
#RGB to YPbPr

img_ypbpr = color.rgb2ypbpr(astronaut_img)

#back to rgb

img_ypbpr_rgb = color.ypbpr2rgb(img_ypbpr)



figure(0)

io.imshow(img_ypbpr)

figure(1)

io.imshow(img_ypbpr_rgb)
io.imsave("this.jpg", img_ypbpr)
from skimage import draw
x, y = draw.line(0, 0, 511, 511)

astronaut_img[x, y] = 0 ## changing the color of the line

# (x, y) are the coordinates of the line

io.imshow(astronaut_img)
#Rectangle example:

def rectangle(x, y, w, h):

    rr, cc = [x, x+w, x+w, x], [y, y, y+h, y+h]

    return (draw.polygon(rr, cc))



rr, cc = rectangle(30, 30, 100, 100)

astronaut_img[rr, cc] = 1

io.imshow(astronaut_img)
#Defining circle coordinates and radius:

x, y = draw.circle(300, 300, 100)

#Draw circle:

astronaut_img[x, y] = 1

#show image:

io.imshow(astronaut_img)
#Defining Bezier Curve coordinates:

x, y = draw.bezier_curve(0,0, 100, 100, 300, 450, 200)

#Drawing Bezier Curve:

astronaut_img[x, y] = 250



io.imshow(astronaut_img)
from skimage import exposure

from pylab import *



img = data.astronaut()



gamma_corrected0 = exposure.adjust_gamma(img, 0.2)

gamma_corrected1 = exposure.adjust_gamma(img, 0.5)

gamma_corrected2 = exposure.adjust_gamma(img, 2)

gamma_corrected3 = exposure.adjust_gamma(img, 5)



figure(0)

io.imshow(gamma_corrected0)

figure(1)

io.imshow(gamma_corrected1)

figure(2)

io.imshow(gamma_corrected2)

figure(3)

io.imshow(gamma_corrected3)
from skimage.transform import rotate

img_rot_180 = rotate(img, 180) #how many degrees to rotate? 180

io.imshow(img_rot_180)
from skimage.transform import resize



img_resized10 = resize(img, (10, 10))

img_resized20 = resize(img, (20, 20))

img_resized40 = resize(img, (40, 40))

img_resized80 = resize(img, (80, 80))

img_resized160 = resize(img, (160, 160))



figure(0)

io.imshow(img_resized10)

figure(1)

io.imshow(img_resized20)

figure(2)

io.imshow(img_resized40)

figure(3)

io.imshow(img_resized80)

figure(4)

io.imshow(img_resized160)
from skimage.measure import compare_ssim as ssim



ssim_original = ssim(img, img, data_range=img.max()-img.min(), multichannel=True)

ssim_different = ssim(img, img_xyz, data_range=(img_xyz.max() - img_xyz.min()), multichannel=True)



print(ssim_original, "/", ssim_different)