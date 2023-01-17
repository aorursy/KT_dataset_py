# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install face_recognition
!pip install imageio-ffmpeg
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import keras

import glob

import cv2

from albumentations import *

from tqdm import tqdm_notebook as tqdm

import gc



from keras.models import Model as KerasModel

from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU

from keras.optimizers import Adam

import face_recognition

import imageio

import tensorflow as tf



import warnings

warnings.filterwarnings('ignore')

PATH = '../input/deepfake-detection-challenge/'

print(os.listdir(PATH))
for dirname, _, filenames in os.walk('/kaggle/input/testmodel'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
TRAIN_PATH = 'train_sample_videos/*.mp4'

TEST_PATH = 'test_videos/*.mp4'

train_img = glob.glob(os.path.join(PATH, TRAIN_PATH))

test_img = glob.glob(os.path.join(PATH, TEST_PATH))
from IPython.display import HTML

from base64 import b64encode

vid1 = open('/kaggle/input/deepfake-detection-challenge/test_videos/ytddugrwph.mp4','rb').read()

data_url = "data:video/mp4;base64," + b64encode(vid1).decode()

HTML("""

<video width=600 controls>

      <source src="%s" type="video/mp4">

</video>

""" % data_url)
class Video:

    def __init__(self, path):

        self.path = path

        self.container = imageio.get_reader(path, 'ffmpeg')

        self.length = self.container.count_frames()

#         self.length = self.container.get_meta_data()['nframes']

        self.fps = self.container.get_meta_data()['fps']

    

    def init_head(self):

        self.container.set_image_index(0)

    

    def next_frame(self):

        self.container.get_next_data()

    

    def get(self, key):

        return self.container.get_data(key)

    

    def __call__(self, key):

        return self.get(key)

    

    def __len__(self):

        return self.length
IMGWIDTH = 256



class Classifier:

    def __init__():

        self.model = 0

    

    def predict(self, x):

        return self.model.predict(x)

    

    def fit(self, x, y):

        return self.model.train_on_batch(x, y)

    

    def get_accuracy(self, x, y):

        return self.model.test_on_batch(x, y)

    

    def load(self, path):

        self.model.load_weights(path)





class Meso4(Classifier):

    def __init__(self, learning_rate = 0.001):

        self.model = self.init_model()

        optimizer = Adam(lr = learning_rate)

        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    

    def init_model(self): 

        x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))

        

        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)

        x1 = BatchNormalization()(x1)

        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        

        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)

        x2 = BatchNormalization()(x2)

        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        

        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)

        x3 = BatchNormalization()(x3)

        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        

        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)

        x4 = BatchNormalization()(x4)

        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        

        y = Flatten()(x4)

        y = Dropout(0.5)(y)

        y = Dense(16)(y)

        y = LeakyReLU(alpha=0.1)(y)

        y = Dropout(0.5)(y)

        y = Dense(1, activation = 'sigmoid')(y)



        return KerasModel(inputs = x, outputs = y)



tf.test.is_gpu_available(

    cuda_only=False,

    min_cuda_compute_capability=None

)
classifier = Meso4()

classifier.load('/kaggle/input/testmodel/Meso4_DF')
submit = []
save_interval = 150 # perform face detection every {save_interval} frames

margin = 0.2

for vi in os.listdir('/kaggle/input/deepfake-detection-challenge/test_videos'):

#     print(os.path.join("/kaggle/input/deepfake-detection-challenge/test_videos/", vi))

    re_video = 0.5

    try:

        video = Video(os.path.join("/kaggle/input/deepfake-detection-challenge/test_videos/", vi))

        re_imgs = []

        for i in range(0,video.__len__(),save_interval):

            img = video.get(i)

            face_positions = face_recognition.face_locations(img)

            for face_position in face_positions:

                offset = round(margin * (face_position[2] - face_position[0]))

                y0 = max(face_position[0] - offset, 0)

                x1 = min(face_position[1] + offset, img.shape[1])

                y1 = min(face_position[2] + offset, img.shape[0])

                x0 = max(face_position[3] - offset, 0)

                face = img[y0:y1,x0:x1]



                inp = cv2.resize(face,(256,256))/255.

                re_img = classifier.predict(np.array([inp]))

    #             print(vi,": ",i , "  :  ",classifier.predict(np.array([inp])))

                re_imgs.append(re_img[0][0])

        re_video = np.average(re_imgs)

        if np.isnan(re_video):

            re_video = 0.5

    except:

        re_video = 0.5

#     submit.append([vi,1.0-re_video])

    submit.append([vi,re_video])
submission = pd.DataFrame(submit, columns=['filename', 'label']).fillna(0.5)

submission.sort_values('filename').to_csv('submission.csv', index=False)

submission