import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample = pd.read_csv("../input/sample_submission.csv")
train = train.iloc[:,0:31]

train.head(10)
test = test.iloc[:,0:30]

test.head(5)
sample.head(5)
from sklearn.preprocessing import LabelEncoder 

labelencoder = LabelEncoder()

train['diagnosis'] = labelencoder.fit_transform(train['diagnosis'])

train.head()
x = train.drop('diagnosis', axis=1)

y = train['diagnosis']
x.head(5)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print(x.shape)

print(y.shape)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import accuracy_score
classify = Sequential()

classify.add(Dense(units=15,activation='relu', input_dim=30))

classify.add(Dense(units=1,activation='sigmoid'))

classify.compile(optimizer='adam',loss="binary_crossentropy",metrics=['binary_accuracy'])
classify.fit(x_train,y_train,batch_size=150,epochs=200)
y_pred = classify.predict(x_test)

y_pred = y_pred > 0.5
acc = accuracy_score(y_test,y_pred)

print("The Accuracy is", acc)