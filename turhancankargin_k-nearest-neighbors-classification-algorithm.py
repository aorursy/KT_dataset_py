# Import important libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plot

from sklearn.model_selection import train_test_split # train test split

from sklearn.neighbors import KNeighborsClassifier # knn model



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

data.head()
data["class"][::20]
A = data[data["class"] == "Abnormal"]

N = data[data["class"] == "Normal"]

# scatter plot

plt.scatter(A.sacral_slope,A.pelvic_incidence,color="red",label="kotu",alpha= 0.3)

plt.scatter(N.sacral_slope,N.pelvic_incidence,color="green",label="iyi",alpha= 0.3)

plt.xlabel("sacral_slope")

plt.ylabel("pelvic_incidence")

plt.legend()

plt.show()
data["class"] = [1 if each == "Abnormal" else 0 for each in data["class"]]

y = data["class"].values

x_data = data.drop(["class"],axis=1)
# normalization 

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
# train test split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
# knn model

knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" k = {} , score = {} ".format(3,knn.score(x_test,y_test)))
score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
knn = KNeighborsClassifier(n_neighbors = 13) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" k = {} , score = {} ".format(13,knn.score(x_test,y_test)))