# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import shutil

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install dlib

!pip install imutils

import numpy as np

import os

import imutils

import dlib # run "pip install dlib"

import cv2 # run "pip install opencv-python"



import imageio

from imutils import face_utils
def rect_to_bb(rect):

    # take a bounding predicted by dlib and convert it

    # to the format (x, y, w, h) as we would normally do

    # with OpenCV

    x = rect.left()

    y = rect.top()

    w = rect.right() - x

    h = rect.bottom() - y



    # return a tuple of (x, y, w, h)

    return (x, y, w, h)



def shape_to_np(shape, dtype="int"):

    # initialize the list of (x, y)-coordinates

    coords = np.zeros((68, 2), dtype=dtype)



    # loop over the 68 facial landmarks and convert them

    # to a 2-tuple of (x, y)-coordinates

    for i in range(0, 68):

        coords[i] = (shape.part(i).x, shape.part(i).y)



    # return the list of (x, y)-coordinates

    return coords
def crop_and_save_image(img, img_path, write_img_path, img_name):

    detector = dlib.get_frontal_face_detector()

    predictor = dlib.shape_predictor('../input/shape-predictor/shape_predictor_68_face_landmarks.dat')

    # load the input image, resize it, and convert it to grayscale



    image = cv2.imread(img_path)

    image = imutils.resize(image, width=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    # detect faces in the grayscale image

    rects = detector(gray, 1)

    if len(rects) > 1:

        print( "ERROR: more than one face detected")

        return

    if len(rects) < 1:

        print( "ERROR: no faces detected")

        return



    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)

        shape = face_utils.shape_to_np(shape)

        name, i, j = 'mouth', 48, 68

        clone = gray.copy()



        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))        

        roi = gray[y:y+h, x:x+w]

        roi = imutils.resize(roi, width = 250, inter=cv2.INTER_CUBIC)        

        print('cropped/' + write_img_path)

        cv2.imwrite('cropped/' + write_img_path, roi)
os.listdir('../input/miraclvc1/dataset')

predictor = dlib.shape_predictor('../input/shape-predictor/shape_predictor_68_face_landmarks.dat')
people = ['F01','F02','F04','F05','F06','F07','F08','F09', 'F10','F11','M01','M02','M04','M07','M08']

data_types = ['words']

folder_enum = ['01','02','03','04','05','06','07','08', '09', '10']

instances = ['01','02','03','04','05','06','07','08', '09', '10']



words = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']          

words_di = {i:words[i] for i in range(len(words))}
if not os.path.exists('cropped'):

    os.mkdir('cropped')
import shutil



def crop_one_person():      

#     if not os.path.exists('cropped'):

#         os.mkdir('cropped')

    people = ['F01','F02','F04','F05','F06','F07','F08','F09', 'F10','F11','M01','M02','M04','M07','M08']

    data_types = ['words']

    folder_enum = ['01','02','03','04','05','06','07','08', '09', '10']

    instances = ['01','02','03','04','05','06','07','08', '09', '10']



    i = 1

    for person_ID in people:

        if not os.path.exists('cropped/' + person_ID ):

            os.mkdir('cropped/' + person_ID + '/')



        for data_type in data_types:

            if not os.path.exists('cropped/' + person_ID + '/' + data_type):

                os.mkdir('cropped/' + person_ID + '/' + data_type)



            for phrase_ID in folder_enum:

                if not os.path.exists('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID):

                    # F01/phrases/01

                    os.mkdir('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID)



                for instance_ID in instances:

                    # F01/phrases/01/01

                    directory = '../input/miraclvc1/dataset/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID + '/'

                    dir_temp = person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID + '/'

    #                 print(directory)

                    filelist = os.listdir(directory)

                    if not os.path.exists('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID):

                        os.mkdir('cropped/' + person_ID + '/' + data_type + '/' + phrase_ID + '/' + instance_ID)



                        for img_name in filelist:

                            if img_name.startswith('color'):

                                image = imageio.imread(directory + '' + img_name)

                                crop_and_save_image(image, directory + '' + img_name,

                                                    dir_temp + '' + img_name, img_name)



#     print(f'Iteration : {i}')

#     i += 1

#     shutil.rmtree('cropped')
# import time



# times = 0

# for _ in range(7):

#     t1 = time.time()

#     crop_one_person()

#     t2 = time.time()

#     times += (t2 - t1)



# print("Average time over 7 iterations : ", times/7)

# crop_one_person()