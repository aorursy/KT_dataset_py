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
import numpy as np

import pandas as pd

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten,BatchNormalization

from keras.optimizers import Adam,Nadam

from sklearn.model_selection import train_test_split

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import shuffle

import matplotlib.pyplot as plt 
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

display(train.shape,test.shape)
Id = test['id']
x_train, x_test, y_train, y_test = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.05)

display(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
x_train, y_train = shuffle(x_train, y_train)
# (60000,784)->(60000,28,28,1)

x_train = x_train.values.reshape(-1,28,28,1)/255.0

x_test = x_test.values.reshape(-1,28,28,1)/255.0



y_train = np_utils.to_categorical(y_train,num_classes=10)

y_test = np_utils.to_categorical(y_test,num_classes=10)
imagegen = ImageDataGenerator(

            rotation_range = 10,     

            width_shift_range = 0.2, 

            height_shift_range = 0.2,         

            shear_range = 20,       

            zoom_range = 0.2  

             )

imagegen.fit(x_train)
model = Sequential()



model.add(Conv2D(input_shape=(28, 28, 1),filters=64, kernel_size=(3, 3), padding='SAME', activation='relu'))

model.add(Conv2D(64, kernel_size=(3, 3), padding='SAME', activation='relu'))

model.add(BatchNormalization(momentum=0.5))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu'))

model.add(BatchNormalization(momentum=0.5))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu'))

model.add(BatchNormalization(momentum=0.5))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(BatchNormalization())

model.add(Dense(10, activation = "softmax"))





model.compile(optimizer=Nadam(),loss='categorical_crossentropy',metrics=['accuracy'])



fit = model.fit_generator(imagegen.flow(x_train,y_train,batch_size=128),epochs=80,validation_data=(x_test,y_test),verbose = 1,steps_per_epoch=100)



loss,accuracy = model.evaluate(x_test,y_test)



print('test loss',loss)

print('test accuracy',accuracy)
plt.figure(figsize=(12,6))

plt.plot(fit.history['loss'])

plt.plot(fit.history['val_loss'])

plt.title('model train vs validation loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')
test = test.drop(['id'],axis=1)

test = test.values.reshape(test.shape[0],28,28,1)/255.0

FINAL_PREDS = model.predict_classes(test)
submission = pd.DataFrame({ 'id': Id,

                            'label': FINAL_PREDS })

submission.to_csv(path_or_buf ="submission.csv", index=False)

submission.head()