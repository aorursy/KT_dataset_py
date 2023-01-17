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
# define the directory where the data is

DATA_DIR = "/kaggle/input/digit-recognizer/"
# load the training samples

train = pd.read_csv(DATA_DIR + "train.csv")



# print out the head of the training sampes

train.head()
# load the testing samples

test = pd.read_csv(DATA_DIR + "test.csv")



test.head()
# prepare the training data into correct format



'''

Take all rows, and all columns (except the first column) to format the training data.

Transform the values into float values

'''

X_train = train.iloc[:, 1:].values

X_train = X_train.astype('float32')

X_train
# prepare the labels for the training set

y_train = train.iloc[:, 0].values.astype('int32')

y_train
X_test = test.values.astype('float32')

X_test
X_train.shape
# convert the training dataset to (# imgs, row_pixels, col_pixels)

X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_train

print(X_train.shape)
import matplotlib.pyplot as plt

%matplotlib inline



for i in range(1, 10):

    # 3 x 3 plots : 3 3 0+i

    plt.subplot(330 + i)

    plt.imshow(X_train[i-1])

    plt.title(y_train[i-1])
# expand 1 more dimension for the grey colour channel

# axis=3 :: expand into 4th dimension

X_train = np.expand_dims(X_train, axis=3)

X_train.shape
# same with X_test

X_test = X_test.reshape(X_test.shape[0], 28, 28)

X_test = np.expand_dims(X_test, axis=3)

X_test.shape
# mean

mean_px = X_train.mean().astype(np.float32)

print(mean_px)



# std: standard deviation

std_px = X_train.std().astype(np.float32)

print(std_px)



def standardize(x):

    return (x-mean_px)/std_px
# one hot encoding is 0 in all dimensions except 1

from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)

print(y_train.shape)

num_classes = y_train.shape[1]

print(num_classes)
plt.title(y_train[0])

plt.plot(y_train[0])

plt.xticks(range(10))
from keras.models import Sequential

from keras.layers import Dense, Lambda, Flatten

from keras.optimizers import Adam



# create a sequential model

model = Sequential()



# standardize and flatten the 28 x 28 input

# element-wise standardization

model.add(Lambda(standardize, input_shape=(28,28,1)))

model.add(Flatten())



# hidden layers with ReLU activation

model.add(Dense(16, input_shape=(784,), activation='relu'))

model.add(Dense(16, input_shape=(16,), activation='relu'))



# output layer with Softmax activation

model.add(Dense(10, input_shape=(16,), activation='softmax'))
# to visualize code



# install ann_visualizer

!pip install ann_visualizer
from ann_visualizer.visualize import ann_viz;



# create a sequential model

model2 = Sequential()



# standardize and flatten the 28 x 28 input

# element-wise standardization

# model2.add(Lambda(standardize, input_shape=(28,28,1)))

# model2.add(Flatten())



# hidden layers with ReLU activation

# model2.add(Dense(16, input_shape=(784,), activation='relu'))

model2.add(Dense(16, input_shape=(16,), activation='relu'))



# output layer with Softmax activation

model2.add(Dense(10, input_shape=(16,), activation='softmax'))



ann_viz(model2, title="vanilla neural network", filename="ann_viz.gv", view=False);
# define the optimizer for the model

adamOptimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)



model.compile(optimizer=adamOptimizer, loss='categorical_crossentropy', metrics=['accuracy'])



print('inputs: ', model.inputs)

print('outputs: ', model.outputs)

print('layers: ', model.layers)

print('summary: ', model.summary())
from sklearn.model_selection import train_test_split

X = X_train

y = y_train



# split the train and validation set in the ratio 4:1

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from keras.preprocessing import image



# use the image data generator to construct the generator

gen = image.ImageDataGenerator()
BATCH_SIZE = 64



# create train, validation data generators so that we don't have to load every training sample

# into the memory at once (which would be very memory extensive)

print(X_train.shape)

train_batches = gen.flow(X_train, y_train, batch_size=BATCH_SIZE)

val_batches = gen.flow(X_val, y_val, batch_size=BATCH_SIZE)

print(train_batches)

print(train_batches.n)
from keras.callbacks import EarlyStopping



earlyStopping = EarlyStopping(monitor='accuracy', patience=8, min_delta=0.01)



# start training the model

history = model.fit_generator(

    generator=train_batches,

    steps_per_epoch=len(X_train)/BATCH_SIZE,

    epochs=20,

    validation_data=val_batches,

    validation_steps=len(X_val)/BATCH_SIZE,

    callbacks=[earlyStopping]

)
plt.title('accuracy vs epochs')

plt.plot(history.history['accuracy'])

plt.show()
X_test_for_display = X_test.reshape(X_test.shape[0], 28, 28)



for i in range(1, 9):

    # 3 x 3 plots : 3 3 0+i

    plt.subplot(330 + i)

    plt.imshow(X_test_for_display[i-1], cmap=plt.get_cmap('gray'))
# predict: returns the scores of the regression

# predict_class: returns the class of the prediction

predictions = model.predict_classes(X_test, verbose=0)

print(predictions)
# create a submissions pandas dataframe 

submissions = pd.DataFrame({

    "ImageId": list(range(1, len(predictions)+1)),

    "Label": predictions

})



# create a csv from the submissions dataframe

submissions.to_csv("VanillaNN.csv", index=False, header=True)