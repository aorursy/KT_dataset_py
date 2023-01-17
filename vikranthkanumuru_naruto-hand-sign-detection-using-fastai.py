from fastai import *

from fastai.vision import *

from tqdm import tqdm_notebook as tqdm

import os

import cv2



import random

import numpy as np

import keras

from random import shuffle

from keras.utils import np_utils

from shutil import unpack_archive

import matplotlib.pyplot as plt

import math

import os

import tensorflow as tf



%reload_ext autoreload

%autoreload 2

%matplotlib inline





video_path="/kaggle/input/test-video-for-testing-naruto-hand-sign-detection/demo input for naruto hand detection.mp4"
!pip install imutils
"""#Preprocess the images such that only the hand sign goes through using opencv



import imutils



# global variables

bg = None



def run_avg(image, aWeight):

    global bg

    # initialize the background

    if bg is None:

        bg = image.copy().astype("float")

        return



    # compute weighted average, accumulate it and update the background

    cv2.accumulateWeighted(image, bg, aWeight)





def segment(image, threshold=25):

    global bg

    # find the absolute difference between background and current frame

    diff = cv2.absdiff(bg.astype("uint8"), image)



    # threshold the diff image so that we get the foreground

    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]



    # get the contours in the thresholded image

    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    # return None, if no contours detected

    if len(cnts) == 0:

        return

    else:

        # based on contour area, get the maximum contour which is the hand

        segmented = max(cnts, key=cv2.contourArea)

        return (thresholded, segmented)







if __name__ == "__main__":

    # initialize weight for running average

    aWeight = 0.5



    # get the reference to the webcam

    camera = cv2.VideoCapture(video_path) 



    # region of interest (ROI) coordinates

    top, right, bottom, left = 10, 350, 225, 590



    # initialize num of frames

    num_frames = 0



    # keep looping, until interrupted

    while(True):

        # get the current frame

        (grabbed, frame) = camera.read()



        # resize the frame

        frame = imutils.resize(frame, width=700)



        # flip the frame so that it is not the mirror view

        frame = cv2.flip(frame, 1)



        # clone the frame

        clone = frame.copy()



        # get the height and width of the frame

        (height, width) = frame.shape[:2]



        # get the ROI

        roi = frame[top:bottom, right:left]



        # convert the roi to grayscale and blur it

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (7, 7), 0)



        # to get the background, keep looking till a threshold is reached

        # so that our running average model gets calibrated

        if num_frames < 30:

            run_avg(gray, aWeight)

        else:

            # segment the hand region

            hand = segment(gray)



            # check whether hand region is segmented

            if hand is not None:

                # if yes, unpack the thresholded image and

                # segmented region

                (thresholded, segmented) = hand



                # draw the segmented region and display the frame

                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                cv2.imshow("Thesholded", thresholded)



        # draw the segmented hand

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)



        # increment the number of frames

        num_frames += 1



        # display the frame with segmented hand

        cv2.imwrite(filename="/kaggle/working/screens/alpha.png", img=clone)



        # observe the keypress by the user

        keypress = cv2.waitKey(1) & 0xFF



        # if the user pressed "q", then stop looping

        if keypress == ord("q"):

            break





cv2.destroyAllWindows()



"""
# When transforming the dataset make sure that the zoom does not exceed too much and the cropping does not crop the hand sign



data = (ImageList.from_folder("/kaggle/input/naruto-hand-sign-dataset/Pure Naruto Hand Sign Data/")

        .split_by_rand_pct()          

        .label_from_folder()

        .add_test_folder() 

        .transform(get_transforms(),size=224)

        .databunch()

        .normalize(imagenet_stats))



#data = (ImageDataBunch.from_folder(mainPath) .random_split_by_pct() .label_from_folder() .transform(tfms, size=224) .databunch())

#data = (ImageList.from_folder(mainPath) .split_by_rand_pct() .label_from_folder() .databunch())



data
data.show_batch(rows=20, figsize=(15,15))
from fastai.metrics import error_rate 



thresh=0.2

learn = cnn_learner(data, models.resnet50 , metrics=[accuracy, error_rate])
#learn.fit_one_cycle(6,1e-2)

#learn.model_dir="/kaggle/working/models"

#learn.save('mini_train')
#learn.fit_one_cycle(6, max_lr=slice(1e-05, 1e-04))

learn.fit_one_cycle(4,1e-2)

learn.model_dir="/kaggle/working/models"

learn.save('Hand-Sign-detection-stage-1')
learn.validate()

learn.show_results(ds_type=DatasetType.Train, rows=3, figsize=(20,20))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,15))
learn.unfreeze() # must be done before calling lr_find

learn.lr_find()

learn.recorder.plot()
#learn.fit_one_cycle(8, max_lr=slice(1e-04, 1e-03))

learn.fit_one_cycle(8,  max_lr=slice(1e-05, 1e-04))
learn.save('Hand-Sign-detection-stage-2')

#learn.export('Hand-Sign-detection-stage-2') 
learn.recorder.plot_losses()
y_preds= learn.get_preds(DatasetType.Test)

y_preds

##getting one image from test dataset and predicting it



#image_data = cv2.imread("/kaggle/input/naruto-hand-sign-dataset/Pure Naruto Hand Sign Data/test/boar/boar_IMG_16519402a-4d5b-11ea-b58b-0242ac1c0002.jpg")



#ying = learn.predict(image_data)
os.listdir("/kaggle/input/test-video-for-testing-naruto-hand-sign-detection") 
learn.load("/kaggle/working/models/Hand-Sign-detection-stage-2")
# classify.py for video processing. 

#############################################################



video_path="/kaggle/input/test-video-for-testing-naruto-hand-sign-detection/demo input for naruto hand detection.mp4"





video_capture = cv2.VideoCapture(video_path) 

#frameRate = video_capture.get(5) #frame rate

i = 0

detected_signs=[]

while True:  # fps._numFrames < 120

    

    ret, frame = video_capture.read() # get current frame

    frameId = video_capture.get(1) #current frame number

    #if (frameId % math.floor(frameRate) == 0):

    if not ret:

        print("End of video file")

        break

    else:  # not necessary

        i = i + 1

        cv2.imwrite(filename="/kaggle/working/screens/"+str(i)+"alpha.png", img=frame); # write frame image to file

        image_data = cv2.imread("/kaggle/working/screens/"+str(i)+"alpha.png")

        pred_category, tensor, probs = learn.predict(image_data)

        #print("{}\n".format(pred.category))

        print ("\n{}\n".format(pred_category))

        ##cv2.imshow("image", frame)  # show frame in window

        #os.remove("/kaggle/working/screens/"+str(i)+"alpha.png")

        cv2.waitKey(1)  # wait 1ms -> 0 until key input





video_capture.release() # handle it nicely
