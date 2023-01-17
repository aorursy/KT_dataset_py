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

        if(filename=="nasa.csv"):

            print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/nasa-asteroids-classification/nasa.csv")

data.columns = [c.replace(' ', '_') for c in data.columns]

data.head()
data.drop(["Neo_Reference_ID","Name","Close_Approach_Date","Epoch_Date_Close_Approach"

           ,"Orbiting_Body","Orbit_Determination_Date","Equinox"],axis=1,inplace=True)

data.Hazardous = [1 if each==True else 0 for each in data.Hazardous]
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler()



y = data.Hazardous.values.reshape(-1,1)

x = data.drop(["Hazardous"],axis=1).values #returns a numpy array

x = scale.fit_transform(x)
c = data.groupby('Hazardous')

a = c.get_group(0) # safe 

b = c.get_group(1) # hazardous



a1, a2 = a.Mean_Anomaly , a.Minimum_Orbit_Intersection

b1, b2 = b.Mean_Anomaly , b.Minimum_Orbit_Intersection
plt.figure(figsize=(6,6))

plt.scatter(a.Mean_Anomaly, a.Minimum_Orbit_Intersection,color="b",label="Safe",alpha=0.1)

plt.scatter(b.Mean_Anomaly,b.Minimum_Orbit_Intersection,color="r", label ="Hazardous",alpha=0.1)

plt.legend()

plt.show()

plt.figure(figsize=(6,6))

plt.scatter(a.Absolute_Magnitude,a.Asc_Node_Longitude,color="b",label="Safe",alpha=0.1)

plt.scatter(b.Absolute_Magnitude,b.Asc_Node_Longitude,color="r", label ="Hazardous",alpha=0.1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train,y_train.ravel())

print("Logistic Regression Acc : ", lr.score(x_test,y_test))
from sklearn.svm import SVC

svc = SVC()

svc.fit(x_train,y_train.ravel())

print("SVM Acc : ",svc.score(x_test,y_test))
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train,y_train.ravel())

print("Decision Tree Acc : ", dtc.score(x_test,y_test))
plt.figure(figsize=(15,15), dpi=100)



tree.plot_tree(dtc,

              feature_names = data.columns,

              rounded = True,

              filled = True,

               class_names = ["Safe","Hazardaus"],

              impurity = True)

plt.savefig("tree.png")