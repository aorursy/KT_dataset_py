# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/test/test"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from PIL import Image
sample_csv = pd.read_csv('../input/sample_submission.csv')

train = pd.read_csv('../input/train.csv')

#test = pd.read_csv('../input/test/test/test.csv')

train.head(10)
figure = plt.figure(figsize=(25, 8))

train_path = "../input/train/train/"

for idx in range(25):

    ax  = figure.add_subplot(5, 25//5, idx+1, xticks=[], yticks=[])

    im = Image.open(f"{train_path}{train.id.iloc[idx]}")

    plt.imshow(im)

    ax.set_title(train.has_cactus.iloc[idx])
train['image'] = pd.DataFrame([Image.open(train_path+x) for x in train.id])
sizes = train.image.map(lambda x : x.size)

sizes.value_counts()
in_shape = np.array(train.image[0]).shape

in_shape
train.head()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import img_to_array

from skimage.io import imread
#add series of images as numpy array

np_imgs = [imread(train_path+x) for x in train.id]

#train['img_array'] = pd.Series([x.reshape(-1, 32, 32, 3) for x in np_imgs])

train['img_array'] = pd.Series([x for x in np_imgs])
for x in train.id[::100]:

    try:

        a = imread(train_path+x)

        #print(a.shape)

    except ValueError:

        print(x)
model = Sequential([

    Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=in_shape, data_format="channels_last"),

    MaxPooling2D(pool_size=(2,2)),

    Dropout(0.25),

    Conv2D(8, kernel_size=(3,3), activation='relu'),

    MaxPooling2D(pool_size=(2,2)),

    Dropout(0.25),

    Flatten(),

    Dense(100, activation='relu'),

    Dense(50, activation='relu'),

    Dense(1,activation='softmax')

])

model.summary()
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
#d = np.array(train.image).reshape(-1,32,32,3)

#d.shape

as_list = np.array([x for x in train.img_array])

as_list.shape
model.fit(as_list, train.has_cactus,

         batch_size = 64,

         epochs=10,

         verbose=2)