# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling

from keras.optimizers import Adam



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#create the training & test sets, skipping the header row with [1:]

train = pd.read_csv("../input/train.csv")

print(train.shape) 

train.head() #앞부분 출력
test = pd.read_csv("../input/test.csv")

print(test.shape)  #배열크기를 출력

test.head()
X_train = (train.ix[:,1:].values).astype('float32')  #all pixel values: 첫번째 열(label)을 빼고 반환

y_train = train.ix[:,0].values.astype('int32')  #only labels i.e targets digits: 첫번째 열(label)만 반환

X_test = test.values.astype('float32')
X_train 
y_train
# 데이터를 keras에 맞게 reshape

# MNIST는 흑백이므로 1개의 채널

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_train.shape 
# X데이터는 흑백이지만, 그 진하기에 따라 0~255까지의 숫자로 되어있음

# 0은 하얀색, 255는 검정색

# 이를 0~1사이의 값으로 Normalize

X_train = X_train.astype('float32')

X_train /= 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_test.shape
X_test = X_test.astype('float32')

X_test /= 255
# y데이터 전처리

from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)

y_train
# 첫번째 convolution layer

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))

print(model.output_shape)
# 두번째 convolution layer 추가 

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(pooling.MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

print(model.output_shape)
# Fully-connected network를 추가

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

print(model.output_shape)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
# 학습

model.fit(X_train, y_train, batch_size=32, nb_epoch=1, verbose=1)
predictions = model.predict_classes(X_test, batch_size=32)

print(predictions)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})

submissions.to_csv("out.csv", index=False, header=True)