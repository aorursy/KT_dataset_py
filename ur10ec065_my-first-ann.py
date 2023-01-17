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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.preprocessing import StandardScaler

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras import backend as K
df = pd.read_csv('/kaggle/input/iris.csv')

df.columns = ['sepal length','sepal width','petal length','petal width','class']

df.head()
#Statistical Summary of dataset

df.describe()



#Checking for missing values



if df.isnull().sum().max() != 0:

    print('Missing Number exists')

else:

    print('No Missing Number')
#Encoding of class variables



df['class'] = df['class'].replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})



#Distribution plots



sns.pairplot(df)
#Correlation matrix



sns.heatmap(df.corr())
#Splitting of data into feature space and target variable



X = df.drop('class',axis = 1)

y = df['class']
#Feature scaling

sc = StandardScaler()

X = sc.fit_transform(X)
#Train-Test split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 22,stratify = y)
#Defining a Neural Network Model with two hidden layers of size 3 dimension each

model = Sequential()

model.add(Dense(3, input_dim=4, activation ="relu"))

model.add(Dense(3, activation = "relu"))

model.add(Dense(3, activation = "softmax"))


def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision
#Defining the loss function, optimizer and meterics

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',precision_m])

y_train = keras.utils.to_categorical(y_train, num_classes=3)

y_test = keras.utils.to_categorical(y_test, num_classes=3)
#Model training

history = model.fit(X_train, y_train,validation_data = (X_test,y_test),epochs=500)
score = model.evaluate(X_test, y_test, batch_size=128)

print('Accuracy of the Model is ',score[1]*100)

print('Precision of the Model is ',score[2]*100)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.plot(history.history['loss']) 

plt.plot(history.history['val_loss']) 

plt.title('Model loss') 

plt.ylabel('Loss') 

plt.xlabel('Epoch') 

plt.legend(['Train', 'Test'], loc='upper left') 

plt.show()