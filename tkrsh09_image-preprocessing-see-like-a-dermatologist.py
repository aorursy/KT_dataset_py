from scipy import ndimage

import operator

import cv2

import numpy as np 

import os 

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt 
image_paths=os.listdir('../input/siim-isic-melanoma-classification/jpeg/train')

image_paths= ["../input/siim-isic-melanoma-classification/jpeg/train/" + str(x) for x in image_paths]
def cv2_clipped_zoom(img, zoom_factor):

    """

    Center zoom in/out of the given image and returning an enlarged/shrinked view of 

    the image without changing dimensions

    Args:

        img : Image array

        zoom_factor : amount of zoom as a ratio (0 to Inf)

    """

    height, width = img.shape[:2] # It's also the final desired shape

    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)



    ### Crop only the part that will remain in the result (more efficient)

    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates

    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2

    y2, x2 = y1 + height, x1 + width

    bbox = np.array([y1,x1,y2,x2])

    # Map back to original image coordinates

    bbox = (bbox / zoom_factor).astype(np.int)

    y1, x1, y2, x2 = bbox

    cropped_img = img[y1:y2, x1:x2]



    # Handle padding when downscaling

    resize_height, resize_width = min(new_height, height), min(new_width, width)

    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2

    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1

    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)



    result = cv2.resize(cropped_img, (resize_width, resize_height))

    result = np.pad(result, pad_spec, mode='constant')

    assert result.shape[0] == height and result.shape[1] == width

    return result
def crop_and_zoom(img):

    bounding=(1024,1024)

    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))

    end = tuple(map(operator.add, start, bounding))

    slices = tuple(map(slice, start, end))

    return cv2_clipped_zoom(img[slices],2)
example_image='../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0368894.jpg'

z=plt.imread(example_image)

plt.imshow(z)
plt.imshow(crop_and_zoom(z))

plt.imsave("example1.png",z)
example_2="../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0094775.jpg"

z2=plt.imread(example_2)

plt.imshow(z2)
plt.imshow(crop_and_zoom(z2))

plt.imsave("example2.png",z2)
example_3="../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0166988.jpg"

z3=plt.imread(example_3)

plt.imshow(z3)
plt.imshow(crop_and_zoom(z3))

plt.imsave("example3.png",z3)
def generate_images(imagelist):

    for x in imagelist:

        img_name=x.split(sep='/')[-1]

        img= plt.imread(x)

        img=crop_and_zoom(img)

        plt.imsave(img_name,img)
generate_images(image_paths[:10])