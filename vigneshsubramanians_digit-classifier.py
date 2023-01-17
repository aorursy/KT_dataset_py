import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train = train.as_matrix()

test = test.as_matrix()
X_train = train[0:25199,1:]

Y_train = train[0:25199,0:1]

x_validate = train[25200:,1:]

y_validate = train[25200:,0:1]
x_train = train[0:,1:]

y_train = train[0:,0:1]

x_test = test[0:,0:]

print(x_train.shape, x_test.shape)
print(X_train.shape,

x_validate.shape)
X_train = X_train.reshape(25199,28,28,1)

x_validate = x_validate.reshape(16800,28,28,1)
from keras.utils import to_categorical
Y_train = to_categorical(Y_train)

y_validate = to_categorical(y_validate)
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten



model = Sequential()



model.add(Conv2D(64,kernel_size = 3, activation = 'relu',input_shape = (28,28,1)))

model.add(Conv2D(32,kernel_size = 3, activation = 'relu'))

model.add(Flatten())

model.add(Dense(10,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.fit(X_train, Y_train, validation_data = (x_validate, y_validate), epochs = 100)
x_test = x_test.reshape(28000,28,28,1)

y_pred_prob = model.predict(x_test[:28000])
y_pred_prob.shape
yt = []

for i in range(len(y_pred_prob)):

    maxi = y_pred_prob[i].argmax()

    yt.append((y_pred_prob[i][maxi],maxi))



yt = np.array(yt)

yt = yt.astype(int)

print(yt)

yt = yt.tolist()

image_id = np.arange(1,len(yt)+1)
label = []

for i in range(len(yt)):

    label.append(yt[i][1])
output = pd.DataFrame({'ImageId': image_id,'Label' : label})

output.to_csv('submission.csv', index=False)