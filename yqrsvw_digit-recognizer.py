# https://github.com/kaggle/docker-python

from keras.layers.convolutional import Conv2D

from keras.layers.core import Dense, Dropout, Flatten, Lambda

from keras.layers.pooling import MaxPooling2D

from keras.models import Sequential

from keras.optimizers import Adam

from keras.utils import np_utils

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.utils import shuffle

from subprocess import check_output

import time
INPUT_PATH = '../input/'

ENCODING = 'utf8'

OUTPUT_PATH = '../working/submission_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'



TRAIN_PATH = INPUT_PATH + '/train.csv'

TEST_PATH = INPUT_PATH + '/test.csv'



TARGET = 'label'



HEIGHT = 28

WIDTH = 28

DEPTH = 1
print(check_output(["ls", INPUT_PATH]).decode(ENCODING))
train = pd.read_csv(TRAIN_PATH)

train.head()
train.describe()
train.shape
plt.hist(train['label'])
train_grouped = train.groupby('label')

digit_labeled = train_grouped.first()
for row in digit_labeled.itertuples():

    digit = np.array(row[1:], dtype=np.uint8)

    digit = digit / 255.0 - 0.5

    digit = digit.reshape(HEIGHT, WIDTH)

    plt.imshow(digit)

    plt.show()
def check_null(data):

    for item in data.columns:

        is_null = data[item].isnull().values.any()



        if is_null:

            print("%s: %s" % (item, is_null))
check_null(train)
shuffle(train)

train_x = train.drop([TARGET], axis=1)

train_y = np_utils.to_categorical(train[TARGET])

classes = train_y.shape[1]
def transform_image(data):

    images = []

    

    for index, row in data.iterrows():

        digit = np.array(row, dtype=np.uint8)

        digit = digit.reshape(HEIGHT, WIDTH, DEPTH)

        images.append(digit)

        

    return np.array(images)
train_x = transform_image(train_x)

train_x.shape
convolution_filter = 32

kernel_size = 5

dropout = 0.2



model = Sequential()

model.add(Lambda(

    lambda x: x / 255.0 - 0.5,

    input_shape=(WIDTH, HEIGHT, DEPTH)

))

model.add(Conv2D(

    convolution_filter,

    (kernel_size, kernel_size),

    activation='relu'

))

model.add(MaxPooling2D())

model.add(Dropout(dropout))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(classes, activation='softmax'))

model.summary()
learning_rate = 1e-6



adam = Adam(lr=learning_rate)

model.compile(optimizer=adam, loss='categorical_crossentropy')

model.fit(train_x, train_y)
test = pd.read_csv(TEST_PATH)

identifier = np.array([x+1 for x in range(test.shape[0])])
test = transform_image(test)

test.shape
probability = model.predict(test)

probability
def maximum(x):

    return list(x).index(max(x))
predictions = list(map(maximum, probability))
output = pd.DataFrame()

output['ImageId'] = identifier

output['Label'] = predictions

output.to_csv(OUTPUT_PATH, index=False)