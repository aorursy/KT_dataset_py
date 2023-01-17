# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img











# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
datagen = ImageDataGenerator(rotation_range=40,

                             width_shift_range=0.2,

                             height_shift_range=0.2,

                             shear_range=0.2,

                             zoom_range=0.2,

                             horizontal_flip=True,

                             vertical_flip=True,

                             fill_mode='nearest')
img = load_img('/kaggle/input/data-improvement-dataset/programming-02.jpg')

x = img_to_array(img)

x = x.reshape((1,) + x.shape)
i = 0



for batch in datagen.flow(x, batch_size=1,

                          save_to_dir='../working/test', 

                          save_format='jpeg'):

    i += 1

    if i > 50:

        break