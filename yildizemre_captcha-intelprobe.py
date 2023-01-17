# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from skimage import io

io.imshow('/kaggle/input/captcha_images_v2/m4fd8.png')
img=io.imread('/kaggle/input/captcha_images_v2/m4fd8.png')
from skimage.filters import threshold_otsu

from skimage.color import rgb2gray

img_gray=rgb2gray(img)

io.imshow(img_gray)
thresh = threshold_otsu(img_gray)

#threeshold uygula emre

binary = img_gray >thresh

io.imshow(binary)
#edge uygula emre 

from skimage.filters import sobel

img_edge=sobel(binary)

io.imshow(img_edge)
#şimdi gaussian

from skimage.filters import gaussian

img_gauss=gaussian(img, multichannel=True)

io.imshow(img_gauss)
#gürültü azalt emre

from skimage.restoration import denoise_tv_chambolle





denoised_image = denoise_tv_chambolle(img, 

                                      multichannel=True)



io.imshow(denoised_image)

io.imshow(img)
from keras import layers

from keras.models import Model

from keras.layers import BatchNormalization
img_shape=(50,200,1)

image = layers.Input(shape=img_shape)

conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(image)

mp1 = layers.MaxPooling2D(padding='same')(conv1)  # 100x25

conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)

mp2 = layers.MaxPooling2D(padding='same')(conv2)  # 50x13

conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)

bn = layers.BatchNormalization()(conv3)

mp3 = layers.MaxPooling2D(padding='same')(bn)

flat = layers.Flatten()(mp3)
outs = []



for i in range(5):

    dens1 = layers.Dense(64, activation='relu')(flat)

    drop = layers.Dropout(0.5)(dens1)

    res = layers.Dense(36, activation='sigmoid')(drop)

    outs.append(res)
model = Model(image, outs)

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
import string

symbols = string.ascii_lowercase + string.digits

num_symbols = len(symbols)

print(num_symbols)
import cv2

n_samples = len(os.listdir('/kaggle/input/captcha_images_v2'))

import cv2

n_samples = len(os.listdir('/kaggle/input/captcha_images_v2'))

X = np.zeros((n_samples, 50, 200, 1)) 

y = np.zeros((5, n_samples, 36)) 



for i, pic in enumerate(os.listdir('/kaggle/input/captcha_images_v2')):

   

    img_gray = cv2.imread(os.path.join('/kaggle/input/captcha_images_v2', pic), cv2.IMREAD_GRAYSCALE)

    pic_target = pic[:-4] 

 

    if len(pic_target) < 6:

       

        img = img_gray / 255.0

        img = np.reshape(img, (50, 200, 1))

       

        targs = np.zeros((5, 36))

        for j, letter in enumerate(pic_target):

            ind = symbols.find(letter)

            targs[j, ind] = 1

       

        X[i] = img 

        y[:, i] = targs

X_train, y_train = X[:970], y[:, :970]

X_test, y_test = X[970:], y[:, 970:]
model.summary()
model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=30,verbose=1, validation_split=0.2)
score= model.evaluate(X_test,[y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])

print('Test Kayıp ve Dogruluk Oranım:', score)