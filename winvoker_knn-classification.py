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
data = pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")

data = pd.DataFrame(data, columns=data.columns)

data
data.income = [1 if each=='>50K' else 0 for each in data.income]

y = data.income

y
data.drop(["fnlwgt","native-country"],axis=1,inplace=True)

data
data = pd.get_dummies(data)

data
over  = data[data.income == 1]

below = data[data.income == 0]

over
plt.scatter(over["capital-gain"],over.income,color='purple',alpha=0.4,label='kotu')

plt.scatter(below["capital-gain"],below.income,color='green',alpha=0.4,label='iyi')
from sklearn.model_selection import train_test_split

x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.neighbors import KNeighborsClassifier

score_list=[]

score_list2=[]

for i in range(1,20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    score_list.append(knn.score(x_test,y_test))

plt.plot(range(1,20),score_list)

plt.show()
score_list2=[]

for i in range(20,40):

    knn2 = KNeighborsClassifier(n_neighbors=i)

    knn2.fit(x_train,y_train)

    score_list2.append(knn2.score(x_test,y_test))

plt.plot(range(20,40),score_list2)

plt.show()
knn3 = KNeighborsClassifier(n_neighbors=33)



knn3.fit(x_train,y_train)

print(knn3.score(x_test,y_test))