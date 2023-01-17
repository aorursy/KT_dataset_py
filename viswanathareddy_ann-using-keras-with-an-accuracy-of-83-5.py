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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



dataset = pd.read_csv("../input/diabetes.csv")

sns.set(style="ticks")

sns.set_palette("husl")

sns.pairplot(dataset.iloc[:,1:8])

plt.show()
dataset.head()
X = dataset.iloc[:,:8]

y = dataset.iloc[:,8]



#Test train split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



#Filling missing values

#Replace zeros with nans

X_train = X_train.replace(0,np.nan)

train_mean = X_train.mean()

#Replace nans with their respective means 

X_train = X_train.fillna(train_mean)

X_test = X_test.replace(0,np.nan)

#Replace nans with their training set mean

X_test = X_test.fillna(train_mean)
X_train = X_train.values

X_test = X_test.values

y_test = y_test.values

y_train = y_train.values



#Standardising the values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
import keras

from keras.models import Sequential

from keras.layers import Dense,Dropout



def build_classifier():

    clf = Sequential()

    clf.add(Dense(units=32, init='uniform', activation='tanh', input_dim=8))

    clf.add(Dropout(0.2))

    clf.add(Dense(units=16, init='uniform', activation='tanh'))

    clf.add(Dropout(0.2))

    clf.add(Dense(units=8, init='uniform', activation='tanh'))

    clf.add(Dense(units=4, init='uniform', activation='tanh'))

    clf.add(Dense(2, activation='softmax'))

    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return clf
clf = build_classifier()

clf.fit(X_train, y_train, epochs=1000)

y_pred = clf.predict(X_test)

y_test_class = np.argmax(y_test,axis=1)

y_pred_class = np.argmax(y_pred,axis=1)



from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test_class,y_pred_class))

print(confusion_matrix(y_test_class,y_pred_class))
scores = clf.evaluate(X_train, y_train, verbose=0)

print ("Training Accuracy")

print("%s: %.2f%%" % (clf.metrics_names[1], scores[1]*100))

scores = clf.evaluate(X_test, y_test, verbose=0)

print ("Testing Accuracy")

print("%s: %f%%" % (clf.metrics_names[1], scores[1]*100))
