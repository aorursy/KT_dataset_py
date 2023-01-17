# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
file=pd.read_csv("../input/Social_Network_Ads.csv")

display(file)
x=file.iloc[:,[2,3]].values

y=file.iloc[:,4].values

x

y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

display(file)
from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

x_train=sc_x.fit_transform(x_train)

x_test=sc_x.fit_transform(x_test)

x_test

x_train
from sklearn.svm import SVC

classifier= SVC(kernel= 'rbf', random_state=0)

classifier.fit(x_train,y_train)
y_pred= classifier.predict(x_test)

y_pred
from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test, y_pred)

cm
from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=classifier, X=x_train,y=y_train,cv=10)

aucc=accuracies.mean()*100

aucc

print('SVM accuracy|'+str(round(aucc,))+'%')