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
# load the data

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test.head()
# check the shape of the train and test

print("Train Shape :" , train.shape)

print("Test Shape  :", test.shape)
train.head()
import matplotlib.pyplot as plt

from matplotlib.pyplot import show

import seaborn as sns

fig = plt.figure(figsize=(12,8))

ax = sns.countplot(x="label", palette="GnBu_d", data=train)

total = float(len(train))

plt.ylabel('Count')

plt.xlabel('Digits')

plt.title('Frquency of each digit')

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height),

            ha="center") 

show()

X_train = train.drop('label',axis=1)

y_train = train['label']

X_train=X_train.values.reshape(-1,28,28,1)

y_train.values.reshape(-1,1)



X_test = test

X_test=X_test.values.reshape(-1,28,28,1)
# normalizing pixel values in range [0,1]

X_train = X_train/255

X_test = X_test/255
W_grid = 4

L_grid = 4



fig, axes = plt.subplots(L_grid, W_grid, figsize = (25, 25))

axes = axes.ravel()



n_training = len(X_train)



for i in np.arange(0, L_grid * W_grid):

    index = np.random.randint(0, n_training) # pick a random number

    axes[i].imshow(X_train[index].reshape([28,28]))

    axes[i].set_title(y_train[index],fontsize=25, color='red')

    axes[i].axis('off')

    

plt.subplots_adjust(hspace = 0.4)

# initialize the input shape

Input_shape = X_train.shape[1:]
# load libraries

import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard,ReduceLROnPlateau
# build the model

cnn_model = Sequential()

cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu', input_shape = Input_shape))

cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))

cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Dropout(0.2))





cnn_model.add(Conv2D(filters = 512, kernel_size = (3,3), activation = 'relu'))

cnn_model.add(Conv2D(filters = 512, kernel_size = (3,3), activation = 'relu'))

cnn_model.add(AveragePooling2D(2,2))

cnn_model.add(Dropout(0.2))



cnn_model.add(Flatten())



cnn_model.add(Dense(units = 1024, activation = 'relu'))



cnn_model.add(Dense(units = 1024, activation = 'relu'))



cnn_model.add(Dense(units = 10, activation = 'softmax'))
cnn_model.summary()
# compile the model

cnn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = keras.optimizers.RMSprop(lr = 0.001), metrics = ['accuracy'])
# fit the model with training data

hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 5, shuffle = True)
# check for the keys to plot

hist.history.keys()
# plot the loss and accuracy

plt.plot(hist.history['loss'])

plt.plot(hist.history['accuracy'])

plt.title('Model Loss vs Accuracy')

plt.ylabel('Value')

plt.xlabel('Epoch Number')

plt.legend(['Training Loss', 'Acccuracy'])
# predicted the labels 

predicted_classes = cnn_model.predict_classes(X_test) 

predicted_classes
# create a submission copy

df_submission = pd.DataFrame([test.index+1,predicted_classes],["ImageId","Label"]).transpose()
# convert to CSV

df_submission.to_csv('digits_submission.csv',index=False)