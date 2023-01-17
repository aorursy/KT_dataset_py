# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")
data.head()
data["class"].value_counts()
data["class"] = ["Normal" if each == "Normal" else "Abnormal" for each in data["class"]]
#done

data["class"].value_counts()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

#x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

x_data = data.drop(["class"],axis=1)

y = data["class"].values
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values
knn.fit(x,y)

prediction = knn.predict(x)

print('Prediction: {}'.format(prediction))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy
knn.fit(x_data,y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_data,y,test_size = 0.3,random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy
# Model complexity

neig = np.arange(1, 31)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for k in range(1,31):

    # k from 1 to 30(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy,color="red", label = 'Testing Accuracy')

plt.plot(neig, train_accuracy,color="blue", label = 'Training Accuracy')

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))