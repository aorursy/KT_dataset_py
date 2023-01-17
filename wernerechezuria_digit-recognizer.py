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
import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split

from keras import  backend as K

from keras.preprocessing.image import ImageDataGenerator



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
[test.shape, train.shape]
# Normalize the data

X_train = train / 255.0

X_test = test / 255.0
X_train = (X_train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train['label'].astype('float32') # only labels i.e targets digits

X_test = X_test.values.astype('float32')
print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28)

print(X_train.shape)



for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i])
plt.subplot(339)

plt.imshow(X_train[11], cmap=plt.get_cmap('gray'))

plt.title(y_train[11])
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_train.shape
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

X_test.shape
mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



def standardize(x): 

    return (x - mean_px)/std_px



[mean_px, std_px]
from keras.utils.np_utils import to_categorical

print(y_train)

y_train= to_categorical(y_train)

num_classes = y_train.shape[1]
num_classes, y_train.shape
plt.title(y_train[9])

plt.plot(y_train[9])

plt.xticks(range(10));
y_train[9]
seed = 43

np.random.seed(seed)

seed
from sklearn.model_selection import train_test_split

from keras.preprocessing import image

gen = image.ImageDataGenerator()



X = X_train

y = y_train

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

batches = gen.flow(X_train, y_train, batch_size=64)

val_batches=gen.flow(X_val, y_val, batch_size=64)

gen =ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,

                               height_shift_range=0.08, zoom_range=0.08)

batches = gen.flow(X_train, y_train, batch_size=64)

val_batches = gen.flow(X_val, y_val, batch_size=64)

from keras.layers.normalization import BatchNormalization

from keras.layers import BatchNormalization, Conv2D , MaxPooling2D



def get_bn_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28,28,1)),

        Conv2D(32,(3,3), activation='relu'),

        BatchNormalization(),

        Conv2D(32,(3,3), activation='relu'),

        MaxPooling2D(),

        BatchNormalization(),

        Conv2D(64,(3,3), activation='relu'),

        BatchNormalization(),

        Conv2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        BatchNormalization(),

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dense(10, activation='softmax')

        ])

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model



model= get_bn_model()

model.optimizer.learning_rate=0.01

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)

model.optimizer.learning_rate=0.01

gen = image.ImageDataGenerator()

batches = gen.flow(X, y, batch_size=64)

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)
predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)
