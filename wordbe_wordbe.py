# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import glob

from matplotlib import pyplot as plt

import cv2

import keras

import random

import json
class_names = sorted([name[:-4] for name in os.listdir('../input/quickdraw-doodle-recognition/train_simplified/')]) # 340 classes

class_paths = sorted(glob.glob('../input/quickdraw-doodle-recognition/train_simplified/' + "*"))



cols=['drawing', 'key_id', 'recognized', 'word']

total_df = pd.DataFrame(columns=cols)

for c in range(len(class_paths)):

    df = pd.read_csv(class_paths[c], usecols=['drawing', 'key_id', 'recognized', 'word'], nrows=1500)

    df = df[df.recognized == True]

    df = df.head(1000)

    df = df.reset_index(drop=True)

    total_df = total_df.append(df, ignore_index=True)

    print(c)

display(total_df)
# 340 * 1000 = 340,000 data divide into 7:3 ratio



from numpy.random import RandomState

rans = RandomState(seed=0)

train = total_df.sample(frac=0.7, random_state=rans)

val = total_df.loc[~total_df.index.isin(train.index)]

train.to_csv('train3401000.csv')

val.to_csv('val3401000.csv')
N_CLASS = 340

BATCH_SIZE = 680

W, H = 256, 256



line_width = 3

class_names = sorted([name[:-4] for name in os.listdir('../input/quickdraw-doodle-recognition/train_simplified/')]) # 340 classes

class_paths = sorted(glob.glob('../input/quickdraw-doodle-recognition/train_simplified/' + "*"))



def load_random_sample(file, sample_size):

    num_lines = sum(1 for l in open(file))



    skip_idx = random.sample(range(1, num_lines), num_lines - (sample_size + 1))

    data = pd.read_csv(file, usecols=['drawing', 'key_id', 'recognized', 'word'], 

                       skiprows=skip_idx)

    return data







def drawing_to_img(drawing_list, line_width):

    img = np.zeros((W, H), np.uint8)

    for x, y in drawing_list:

        for i in range(len(x) - 1):

            cv2.line(img, (x[i], y[i]), (x[i+1], y[i+1]), 1, line_width)

    

    return img







def generator(line_width):

    sample_size = BATCH_SIZE // N_CLASS

    

    while True:

        img_batch = np.zeros((BATCH_SIZE, W, H, 1))

        label_batch = []



        i = 0

        for file in class_csvs:

            df = load_random_sample(file, sample_size)

            df['drawing'] = df['drawing'].apply(json.loads)

            

            for one_drawing in df.drawing.values:

                img_batch[i,:,:,0] = drawing_to_img(one_drawing, line_width)

                i += 1

            

        return img_batch



            



gen = generator(3)

# x = next(gen)



for i in gen:

    plt.imshow(i[:,:,0])

    plt.show()

    

#     label_batch = keras.utils.to_categorical(df['word'])

#             plt.imshow(img_batch[i,:,:,0])

#             plt.show()
from keras.applications.mobilenetv2 import MobileNetV2