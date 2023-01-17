from IPython.display import Image
Image("../input/Dance_Robots_Comic.jpg")
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import skimage
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.util import crop, pad
from skimage.morphology import label
from skimage.color import rgb2gray, gray2rgb

import os

import zipfile
z = zipfile.ZipFile("Dancer_Images.zip", "w")
cap = cv2.VideoCapture('../input/Shadow Dancers 1 Hour.mp4')
print(cap.get(cv2.CAP_PROP_FPS))
%%time

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
count = 0
TRAIN_SIZE = 27000
FRAME_SKIP = 2
IMG_WIDTH = 96
IMG_HEIGHT = 64
IMG_CHANNELS = 1

video = cv2.VideoWriter('Simple_Shadow_Dancer_Video.avi',cv2.VideoWriter_fourcc(*"MJPG"), 30, (IMG_WIDTH, IMG_HEIGHT), False)

while(count < TRAIN_SIZE):
    try:
        ret, frame = cap.read()

        if currentFrame % FRAME_SKIP == 0:
            count += 1
            if count % int(TRAIN_SIZE/10) == 0:
                print(str((count/TRAIN_SIZE)*100)+"% done")
            # preprocess frames
            img = frame
            img = rgb2gray(img)
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
            img[img > 0.2] = 255
            img[img <= 0.2] = 0
            # save frame to zip and new video sample
            name = './data/frame' + str(count) + '.jpg'
            cv2.imwrite(name, img)
            video.write(img.astype('uint8'))
            z.write(name)
            os.remove(name)
    except:
        print('Frame error')
        break
    currentFrame += 1

print(str(count)+" Frames collected")
cap.release()
z.close()
video.release()

cap.release()
z.close()
video.release()