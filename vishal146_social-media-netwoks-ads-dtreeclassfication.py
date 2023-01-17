# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/social-media-networks/Social_Network_Ads.csv")
data.head()
#features of the dataset -- age and EstimatedSalary

#target feature of the dataset -----Purchased

feature_coln=['Age','EstimatedSalary']

X=data[feature_coln]

y=data['Purchased']
X.head()

y.head()
#splitting data into training and testing

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25, random_state=0)
X_train.head()
from sklearn.preprocessing import StandardScaler

X_train=StandardScaler().fit_transform(X_train)

X_test=StandardScaler().fit_transform(X_test)



#model creation and training

from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

model.fit(X_train,y_train)
#prediction from the model 

pred=model.predict(X_test)
pred
#Evaluating Algo

#Accuracy

from sklearn import metrics

print('Accuracy of the model = ',metrics.accuracy_score(y_test,pred))
#Evaluating  algo using confusion matrix

from sklearn.metrics import confusion_matrix

print('Confusion matrix of the model = ',confusion_matrix(y_test,pred))

from sklearn import tree 

tree.plot_tree(model.fit(X_train,y_train))

#Decision Tree Visualization 

from sklearn.tree import export_graphviz

import graphviz

dot_data = tree.export_graphviz(model, out_file=None, 

                      feature_names=feature_coln,  

                      class_names=None,  

                      filled=True, rounded=True,  

                      special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
