# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/column_2C_weka.csv",sep=",")
data.info()
data.head()
# "class" is a special name for pyhton because it is a structure in commands for this reason we change class by "classs""

data.columns = ["classs" if each=="class" else each for each in data.columns]

data.head()
# Seperating data to 2 dataframes which are N (Normal) and A (Abnormal)

N = data[data.classs == "Normal"]

A = data[data.classs == "Abnormal"]
# corelation map

f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True, linewidths=.5, fmt='.2f', ax=ax)
# scatter plot

plt.scatter(A.sacral_slope,A.pelvic_incidence,color="red",label="Abnormal")

plt.scatter(N.sacral_slope,N.pelvic_incidence,color="green",label="Normal")

plt.xlabel("sacral_slope")

plt.ylabel("pelvic_incidence")

plt.legend() #labelleri gostermeye yarar

plt.show()

data.classs =[1 if each == "Abnormal" else 0 for each in data.classs]
data
# we seperate data to x_data and y.

y = data.classs.values

x_data = data.drop(["classs"],axis=1)

# normalization

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

# Create KNN model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3) #n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

# score

print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))
# Finding optimum k value between 1 and 15

score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each) # create a new knn model

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))



plt.plot(range(1,15),score_list) # x axis is in interval of 1 and 15

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()

# finding max value in a list and it's index.

a = max(score_list) # finding max value in list

b = score_list.index(a)+1 # index of max value.

print("k = ",b," and maximum value is ", a)