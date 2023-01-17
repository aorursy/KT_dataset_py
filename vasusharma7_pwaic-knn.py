import numpy as np

import pandas as pd

dt = pd.read_csv("../input/training.csv")
dt.head()
dt["area_sepal"] = dt["sepal_length"]*dt["sepal_width"]
dt["area_petal"] = dt["petal_length"]*dt["petal_width"]
dt.head()
from sklearn.model_selection import train_test_split as tts
Y  = dt.pop("label")

X = dt
x_train , x_test,y_train,y_test  = tts(X,Y,random_state = 1,test_size=0.2)
K = np.arange(1,5)
acc = []

from sklearn.neighbors import KNeighborsClassifier

for i in K:

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    acc.append(knn.score(x_test,y_test))

acc
from matplotlib import pyplot as plt

plt.plot(K,acc)
k = 3

knn = KNeighborsClassifier(3)

knn.fit(x_train,y_train)
predicted = knn.predict(x_test)
test = pd.read_csv("../input/testing.csv")

test["area_sepal"] = test["sepal_length"]*test["sepal_width"]

test["area_petal"] = test["petal_length"]*test["petal_width"]
output = knn.predict(test)
index = list(np.arange(1,test.shape[0]+1))
import csv

with open("submit1.csv" , "w",newline="") as file:

    writer = csv.writer(file,lineterminator = "\n")

    writer.writerow(["id","label"])

    for i,j in enumerate(index):

        writer.writerow([j,output[i]])