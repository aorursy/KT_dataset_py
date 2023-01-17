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
!pip install tensorflow
!conda list tensorflow
from tqdm import tqdm

# from tqdm import tqdm_notebook as tqdm # Jupyter Notebookでtqdmを使用する場合

import numpy as np

import pandas as pd

from PIL import Image

import os

import gc

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

from keras.utils.np_utils import to_categorical
np.set_printoptions(suppress=True)
def convert_and_resize(img, to_array=True, color_mode='L', size=(256, 256)):

    img_ = img.resize(size)

    img_ = img_.convert(color_mode)

    if to_array:

        img_ = np.array(img_).astype(np.float32) / 255.

    return img_
img_size = (256, 256)
X_train = []

y_train = []

print('Loading NORMAL Training Data...')

for i in range(1,7):

    for file_name in tqdm(sorted(os.listdir('../input/1056lab-defect-detection-extra/train/Class'+str(i)))):

        X_train.append(convert_and_resize(Image.open(os.path.join('../input/1056lab-defect-detection-extra/train/Class'+str(i), file_name)),

                                         size=img_size))

        y_train.append(0)



print('Loading DEFECT Training Data...')

for i in range(1,7):

    for file_name in tqdm(sorted(os.listdir('../input/1056lab-defect-detection-extra/train/Class'+str(i)+'_def'))):

        X_train.append(convert_and_resize(Image.open(os.path.join('../input/1056lab-defect-detection-extra/train/Class'+str(i)+'_def', file_name)),

                                          size=img_size))

        y_train.append(1)

print('Done.')
X_test = []

print('Loading Test Data...')

for file_name in tqdm(sorted(os.listdir('../input/1056lab-defect-detection-extra/test'))):

    X_test.append(convert_and_resize(Image.open(os.path.join('../input/1056lab-defect-detection-extra/test', file_name)),

                                    size=img_size))

print('Done')
X_train = np.array(X_train).reshape(len(X_train), img_size[0], img_size[1], 1) 

X_test = np.array(X_test).reshape(len(X_test), img_size[0], img_size[1], 1)
X_train.shape, X_test.shape
y_train = np.array(y_train).reshape(-1, 1)

y_train = to_categorical(y_train, 2)
p = np.random.permutation(len(X_train))

X_train = X_train[p]

y_train = y_train[p]
model = Sequential()



model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((3, 3)))



model.add(Flatten()) 

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(2, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.3, batch_size=32, epochs=10)
predict = model.predict_proba(X_test, batch_size=32)[:, 1]
submit = pd.read_csv('../input/1056lab-defect-detection-extra/sampleSubmission.csv')

submit['defect'] = predict

submit.to_csv('submission.csv', index=False)