# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from keras.utils.np_utils import to_categorical

img_size = (224,224)

img_array_list = []

cls_list = []

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/0/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(0)



img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/1/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(1)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/2/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(2)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/3/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(3)



img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/4/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(4)



img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/5/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(5)



img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/6/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(6)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/7/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(7)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/8/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(8)

    

img_list1 = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/train/9/*.jpg')

for i in img_list1:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)

    cls_list.append(9)

    

X_train = np.array(img_array_list)

y_train = to_categorical(np.array(cls_list))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from tensorflow.keras.metrics import AUC

#Conv2D 畳み込み計算



model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(rate=0.1))



model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(rate=0.1))





model.add(Flatten()) #特徴を一次元に

model.add(Dense(units=256, activation='relu',input_dim=2))

model.add(Dense(units=256, activation='relu'))

model.add(Dropout(rate=0.1))



model.add(Dense(units=10, activation='softmax'))





auc = AUC()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', auc])

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train,epochs=49, batch_size=160)
import glob

from keras.preprocessing.image import load_img, img_to_array



img_array_list = []



img_list = glob.glob('/kaggle/input/1056lab-archit-heritage-elem-recognit/test/*.jpg')

img_list.sort()

for i in img_list:

    img = load_img(i, color_mode='grayscale', target_size=(img_size))

    img_array = img_to_array(img) / 255

    img_array_list.append(img_array)



X_test = np.array(img_array_list)
predict = model.predict(X_test)[:, 1]



submit = pd.read_csv('/kaggle/input/1056lab-archit-heritage-elem-recognit/sampleSubmission.csv')

submit['class'] = predict

submit.to_csv('submission.csv', index=False)
submit