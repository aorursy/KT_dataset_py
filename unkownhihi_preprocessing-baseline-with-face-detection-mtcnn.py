!pip install ../input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from time import time

import os

from mtcnn import MTCNN

from tqdm import tqdm_notebook

import time

import gc

import random
train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'

train_video_files = [train_dir + x for x in os.listdir(train_dir)]

test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'

test_video_files = [test_dir + x for x in os.listdir(test_dir)]
import pandas as pd

df_train = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json').transpose()

df_train=dict(df_train)['label']

df_train.head()
LABELS = ['FAKE','REAL']
detector = MTCNN()
THRESHOLD=0.7

def detect_face(img,ratio):

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    final = []

    detected_faces_raw = detector.detect_faces(img)

    if detected_faces_raw==[]:

        print('no faces found')

        return []

    for x in detected_faces_raw:

        if x['confidence']<THRESHOLD:

            continue

        x,y,w,h=x['box']

        #x,y,w,h=ratio*x,ratio*y,ratio*w,ratio*h

        final.append([x,y,w,h])

    return final
RESIZING_RATIO=2

FACES_TAKE = 1

def detect_video(video):

    face_frames=[]

    cap = cv2.VideoCapture(video)

    cap.set(cv2.CAP_PROP_FPS, 24)

    ret,frame = cap.read()

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    t=tqdm_notebook(total=total)

    while True:

        t.update()

        #cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))   

        ret,frame = cap.read()

        if ret!=True:

            break

        frame=cv2.resize(frame,(int(1920/RESIZING_RATIO),int(1024/RESIZING_RATIO)))

        faces = detect_face(frame,RESIZING_RATIO)

        if faces==[]:

            face_frames.append([])

            continue

        if FACES_TAKE==1:

            x,y,w,h=faces[0]

            face_frames.append(frame[y:h+y,x:w+x])

            continue

        face_frames.append([])

        count=0

        for (x,y,w,h) in faces:

            count+=1

            croped_face = frame[y:h+y,x:w+x]

            face_frames[-1].append(croped_face)

            if count==FACES_TAKE:

                break

    return face_frames
def video(video):

    cap = cv2.VideoCapture(video)

    cap.set(cv2.CAP_PROP_FPS, 24)

    ret,frame = cap.read()

    final=[]

    while True:

        if ret!=True:

            break

        ret,frame = cap.read()

        final.append(frame)

    return final
def random_choice(face_frames,num):

    return [random.choice(face_frames) for _ in range(num)]
def show(face_frames,num_faces):

    if FACES_TAKE==1:

        for x in random_choice(face_frames,num_faces):

            try:

                if type(x)!=np.ndarray:

                    x=np.zeros(face_frames[0].shape)

                else:

                    x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)

                plt.imshow(x)

                plt.show()

            except:

                pass

    else:

        pass #currently working on it
face_frames=detect_video(train_video_files[0])
show(face_frames,5)
gc.collect()
def pipeline(video_files):

    X_train=[]

    y_train=[]

    for video_file in video_files:

        X=detect_video(video_file)

        y=labels.index(df_train[video_files])

        X_train.append(X)

        y_train.append(y)

    return X_train,y_train
#pipeline(train_video_files)
submission = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

submission['label'] = 0.45

submission.to_csv('submission.csv', index=False)