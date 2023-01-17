# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D,Activation, Dropout, Flatten, Dense

from keras.utils import to_categorical

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
def split_train_test(trian_file,test_file,image_dim):

    train_valid_df=pd.read_csv(trian_file)

    test_df=pd.read_csv(test_file)

    train_df,valid_df=train_test_split(train_valid_df,

                                       test_size=0.2, 

                                       random_state=13,

                                       stratify=list(train_valid_df['label']))

    train_pixels=train_df.iloc[:,1:].as_matrix()

    valid_pixels=valid_df.iloc[:,1:].as_matrix()

    test_pixels=test_df.iloc[:,1:].as_matrix()

    train_labels=to_categorical(train_df['label'])

    valid_labels=to_categorical(valid_df['label'])

    test_labels=to_categorical(test_df['label'])

    train_greys=train_pixels.reshape(train_pixels.shape[0],image_dim[0],image_dim[1],1)

    valid_greys=valid_pixels.reshape(valid_pixels.shape[0],image_dim[0],image_dim[1],1)

    test_greys=test_pixels.reshape(test_pixels.shape[0],image_dim[0],image_dim[1],1)

    return train_greys,train_labels,valid_greys,valid_labels,test_greys,test_labels
train_greys,train_labels,valid_greys,valid_labels,test_greys,test_labels=split_train_test(

    '../input/fashion-mnist_train.csv',

    '../input/fashion-mnist_test.csv',[28,28])
train_greys = train_greys.astype('float32')

test_greys = test_greys.astype('float32')

valid_greys = valid_greys.astype('float32')

train_greys /= 255

test_greys /= 255

valid_greys /= 25
model=Sequential()

model.add(Conv2D(32, (3, 3),input_shape=(28,28,1),kernel_initializer='he_normal',activation='relu', padding='same', name='block1_conv1'))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2'))

model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2'))

model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

model.add(Flatten(name='flatten'))

model.add(Dropout(0.05))

model.add(Dense(128,activation='relu',name='dense'))

model.add(Dense(10,activation='softmax',name='prediction'))

model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])
model.fit(train_greys,train_labels,batch_size=128,epochs=10,verbose=1,validation_data=(valid_greys,valid_labels))
score=model.evaluate(test_greys,test_labels,verbose=0)
print(score[1])