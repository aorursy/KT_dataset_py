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
import pandas as pd

mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

mnist_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
print(mnist_train.shape)

print(mnist_test.shape)
#First column of the dataset has lables so all rows and column till index 1



train_lables = mnist_train.iloc[:, :1].values

test_lables = mnist_test.values[:, :1]

#Drop Lable column from data

train_images = mnist_train.drop('label', axis=1)

test_images = mnist_test.drop('label', axis=1)

#Encode the lables using one hot encoder

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

y_train = ohe.fit_transform(train_lables).toarray()

y_test = ohe.fit_transform(test_lables).toarray()
#Import Sequential Model and Dense layes for Connect Network



from keras.models import Sequential

from keras.layers import Dense



#Add and configure hidden layers

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(784, )))

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))



#Add optimizer, loss function and matrics to compile mode

#We will use Categorical Cross Entropy because we have more than 2 output class

#If output class are 2 or less then we will use Binary Cross Entropy

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



    
#Train The Model



history = model.fit(train_images, y_train, epochs=5, batch_size=60)
#Test the model

model.evaluate(test_images, y_test)
model.save_weights('model.h5')
#Use trained model to make predictions

pred = model.predict(test_images[0:4])
print(np.argmax(pred, axis=1))
print(test_lables[0:1])
import matplotlib.pyplot as plt



first_image = test_images.values[0:1,:]

first_image = np.array(first_image, dtype='float')

pixels = first_image.reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()

