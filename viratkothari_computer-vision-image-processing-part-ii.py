import numpy as np

import cv2

import matplotlib as mpl

import matplotlib.pyplot as plt
# Loading image



path = "../input/images/8.png"

img=plt.imread(path)

print(img.shape) # printing shape of image. It is colour so 124 x 408 pixel having 3 channel - RGB

#print(img) # printing array value of image

plt.imshow(img)

plt.title("Original Image")

plt.show() # printing image
# Erosion: loss of pixles from the edges



kernel = np.ones((7,7)) # we will lose 5 X 5 pixels from surronding pixels of the each object of the image

#kernel #printing array



img2=cv2.erode(img,kernel)



print(img.shape) # printing shape of image. It is colour so 124 x 408 pixel having 3 channel - RGB

#print(img) # printing array value of image

plt.imshow(img2)

plt.title("Erosion Output")

plt.show() # printing image. Small objects/noice are removed and objects become thiner
# Dilation: gain of pixles on the edges



kernel = np.ones((7,7)) # we will gain 5 X 5 pixels on surronding pixels of the each object of the image

#kernel #printing array



img3=cv2.dilate(img2,kernel) # we are using the image on which we already applied Erosion



print(img.shape) # printing shape of image. It is colour so 124 x 408 pixel having 3 channel - RGB

#print(img) # printing array value of image

plt.imshow(img3)

plt.title("Dilation Output")

plt.show() # printing image. The objects become more thicker

# Apply Erosion and Dilation on half of the image



img4=img.copy()

img4[:,:204]=cv2.dilate(cv2.erode(img4[:,:204],kernel),kernel)



print(img.shape) # printing shape of image. It is colour so 124 x 408 pixel having 3 channel - RGB

plt.imshow(img4)

plt.title("Erosion and Dilation on half of the image Output")

plt.show()
# Inversion of the image



img5=abs(255-np.uint8(img4))

print(img5.shape) # printing shape of image. It is colour so 124 x 408 pixel having 3 channel - RGB

plt.imshow(img5)

plt.title("Image Inversion Output")

plt.show()
# Image Rotation



img6=cv2.rotate(img,cv2.ROTATE_180) # Other options: ROTATE_90_CLOCKWISE and ROTATE_90_ANTICLOSKWISE

print(img6.shape)

plt.imshow(img6)

plt.title("Image Rotation at 180 degree output")

plt.show()
# Image Flipping - Verticle



img7=cv2.flip(img,0) # 0 is for verticle flipping

print(img7.shape)

plt.imshow(img7)

plt.title("Image Flipping - Verticle")

plt.show()
# Image Flipping - Horizontal



img8=cv2.flip(img,1) # 1 is for horizontal flipping

print(img8.shape)

plt.imshow(img8)

plt.title("Image Flipping - Horizontal")

plt.show()
# Image Flipping - Verticle and Horizontal both



img9=cv2.flip(img,-1) # -1 is for verticle and horizontal both flipping

print(img9.shape)

plt.imshow(img9)

plt.title("Image Flipping - Verticle and Horizontal")

plt.show()
print("Notebook completed!")