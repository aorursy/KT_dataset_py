import numpy as np

import  seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

from matplotlib import pyplot as plt

df = pd.read_csv('../input/train.csv')

df
output = np.array(df['label'])

df = df.drop(['label'],axis = 1)

y_train = []

for each in output:

    lst = [0 for _ in range(0,10)]

    lst[each] = 1

    y_train.append(lst)

y_train = np.array(y_train)

y_train.shape
img = np.array(df.iloc[9])

img = np.reshape(img,(28,28))

plt.imshow(img, interpolation='nearest')

plt.show()
x_train = []

for i in range (0,42000):

    img = np.array(df.iloc[i])

    img = np.reshape(img,(28,28))

    x_train.append(img)

x_train  = np.array(x_train)

x_train.shape
x_train = np.reshape(x_train,(42000, 28, 28, 1))

x_train.shape
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,

          batch_size=128,

          epochs=12,

          verbose=1,

          validation_split=(0.3))
df = pd.read_csv('../input/test.csv')

x_test = []

for i in range (0,len(df)):

    img = np.array(df.iloc[i])

    img = np.reshape(img,(28,28))

    x_test.append(img)

x_test  = np.array(x_test)

x_test = np.reshape(x_test,(len(df), 28, 28, 1))

pred = model.predict(x_test)

result = []

for each in pred:

    for i in range (0,len(each)):

        if(each[i] == max(each)):

            result.append(i)

ImageId = [i+1 for i in range(0,len(df))]

Label = result

dic = {'ImageId':ImageId,'Label':Label}

df_result = pd.DataFrame(dic)

df_result.to_csv('submission.csv', index=False)
img = np.array(df.iloc[23])

img = np.reshape(img,(28,28))

plt.imshow(img, interpolation='nearest')

plt.show()

print(result[23])