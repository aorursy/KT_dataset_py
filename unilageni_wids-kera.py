# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os







# Any results you write to the current directory are saved as output.
from PIL import Image

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np 

from glob import glob

import keras

from keras.models import Sequential

from keras.layers import *

from keras.optimizers import Adam, SGD, RMSprop,Adadelta
train = pd.read_csv('../input/traininglabels.csv')
train.head()
X=train.image_id

y=train.has_oilpalm
X.head()
train_images=[]

train_labels=[]

test_images=[]

test_labels=[]
split=StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)



# #Train

for train_index,test_index in split.split(X,y):

    for i in train_index:

        image=Image.open('../input/train_images/train_images/'+X.loc[i]).convert('L').resize((220,220))

        image=np.asarray(image)/255

        train_images.append(image)

        train_labels.append(y.loc[i])



    for i in test_index:

        image=Image.open('../input/train_images/train_images/'+X.loc[i]).convert('L').resize((220,220))

        image=np.asarray(image)/255

        test_images.append(image)

        test_labels.append(y.loc[i])
test_images[0]
train_images=np.array(train_images)

test_images=np.array(test_images)
train_images=train_images.reshape((-1,220,220,1))

test_images=test_images.reshape((-1,220,220,1))
test_labels=keras.utils.to_categorical(test_labels)

train_labels=keras.utils.to_categorical(train_labels)
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range = 40,

                            width_shift_range=0.2,

                            height_shift_range=0.2,

                            shear_range=0.2,

                            zoom_range=0.2,

                            horizontal_flip=True,

                            fill_mode='nearest')
from keras.models import Sequential

from keras import models,layers
256*2
from keras import layers

from keras import models

from keras import regularizers



model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation='relu',

                        kernel_regularizer=regularizers.l2(0.001),

                        input_shape=(220, 220, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.5))





model.add(layers.Conv2D(32, (3, 3), activation='relu',kernel_regularizer=regularizers.l1(0.001)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.5))



model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.5))



model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.5))





model.add(layers.Conv2D(256, (5, 5), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.5))



model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1024, activation = 'relu'))

model.add(layers.Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

history = model.fit(train_images,train_labels,verbose=1,

         batch_size=train_images.shape[0]//128,    

          epochs=20,

          validation_data=(test_images,test_labels),

          )
import matplotlib.pyplot as plt

%matplotlib inline

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()

from glob import glob

test_data=[]

test_id=[]

for i in glob('../input/leaderboard_test_data/leaderboard_test_data/*'):

    image=Image.open(i).convert('L').resize((220,220))

    image=np.asarray(image)/255

    test_data.append(image)

    test_id.append(i)

    
holdout_data=[]

holdout_image_id=[]

for i in glob('../input/leaderboard_holdout_data/leaderboard_holdout_data/*'):

    image=Image.open(i).convert('L').resize((220,220))

    image=np.asarray(image)/255

    holdout_data.append(image)

    holdout_image_id.append(i)
test_data=np.array(test_data)

test_data=test_data.reshape((-1,220,220,1))

holdout_data=np.array(holdout_data)

holdout_data=holdout_data.reshape((-1,220,220,1))



prediction1=model.predict(test_data)

prediction2=model.predict(holdout_data)
prediction1


predictions=np.concatenate((prediction1,prediction2))
predictions
predictions=np.argmax(predictions,axis=-1)
predictions
image_id=[i[-17:] for i in test_id]+[i[-17:] for i in holdout_image_id]
submission=pd.DataFrame({'image_id':image_id,'has_oilpalm':predictions})
submission.to_csv('smalls.csv',index=False)
submission.has_oilpalm.value_counts()