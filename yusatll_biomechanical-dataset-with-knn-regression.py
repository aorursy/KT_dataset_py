# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

data.info()
data.head(10)
# x is all data except 'class' column

x = data.loc[:,data.columns != 'class']

print(x)
# y is only 'class' column

y = data['class']

print(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)



#y_train = y_train.reshape(-1,1)

#y_test = y_test.reshape(-1,1)



print("x_train shape: ",x_train.shape)

print("x_test shape: ",x_test.shape)

print("y_train shape: ",y_train.shape)

print("y_test shape: ",y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

# predict

knn.fit(x,y)

predicts = knn.predict(x)

print("Predicts: ", predicts)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

prediction = knn.predict(x_test)

print("KNN K = {} accuracy: {}".format(5, knn.score(x_test, y_test)))
test_list = []

train_list = []

for k in range(1,30):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train, y_train)

    train_list.append(knn.score(x_train, y_train))

    test_list.append(knn.score(x_test, y_test))



# Graph

plt.figure()

plt.plot(range(1,30), test_list, label = "Test Accuracies")

plt.plot(range(1,30), train_list, label = "Train Accuracies")

plt.xlabel("Neighbors")

plt.ylabel("Accuracies")

plt.show()



print("Best accuracy: {} and number of neighbors: {}".format(np.max(test_list), test_list.index(np.max(test_list))+1))
data.head()
# data create

data1 = data[data['class'] =='Abnormal']

x =  np.array(data1.loc[:,'lumbar_lordosis_angle']).reshape(-1,1)

y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
print(min(x),max(x), min(y), max(y))
# show labels

plt.figure(figsize=[8,8])

plt.scatter(x=x,y=y)

plt.xlabel('lumbar_lordosis_angle')

plt.ylabel('sacral_slope')

plt.show()
# Linear Regression Model

from sklearn.linear_model import LinearRegression

lr = LinearRegression()



pre = np.linspace(min(x), max(x)).reshape(-1,1)



lr.fit(x,y)



predicted = lr.predict(pre)



print("R_square : ",lr.score(x,y))
plt.plot(pre, predicted, color='red', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('lumbar_lordosis_angle')

plt.ylabel('sacral_slope')

plt.show()