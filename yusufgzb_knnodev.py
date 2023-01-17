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
data2=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

data2.info()
data2["class"] = [1 if each=="Abnormal" else 0 for each in data2["class"]]

y=data2["class"].values

xData=data2.drop(["class"],axis=1)
x=(xData-np.min(xData))/(np.max(xData)-np.min(xData))
from sklearn.model_selection import train_test_split

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size = 0.3,random_state=42)
from sklearn.neighbors import KNeighborsClassifier

score=[]

for each in range(1,10):

    

    knn2=KNeighborsClassifier(n_neighbors = each)

    knn2.fit(xTrain,yTrain)

    

    score.append(knn2.score(xTest,yTest))

    

plt.plot(range(1,10),score)

plt.show()





knn = KNeighborsClassifier(n_neighbors = 7) 

knn.fit(xTrain,yTrain)

prediction = knn.predict(xTest)

print(prediction)