# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#loading the Data

df = pd.read_csv("../input/HR_comma_sep.csv")
df.head()

#df[df["sales"]!= "sales"]
#plt.figure(figsize=(12,6))

sns.pairplot(df, palette='husl',hue = 'left')
#plt.figure(figsize=(15,5))

sns.jointplot('number_project','satisfaction_level',data = df, kind = 'kde',size= 10)
df.head()
# Getting dummi variables to update for the sales and salary columns

newDF = pd.get_dummies(df,columns=["sales",'salary'],drop_first=True)
(newDF.head())

#print (len(newDF.columns))
# importing few things

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.svm import SVC

from sklearn.grid_search import GridSearchCV
X = newDF.drop('left',axis =1)

y = newDF['left']
#spliting training and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.65, random_state=101)
svm = SVC(C=100,kernel='rbf',gamma=0.01, verbose=3)
svm.fit(X_train,y_train)
pred = svm.predict(X_test)
print (confusion_matrix(pred,y_test))

print (classification_report(pred,y_test))
#using grid Search method to find best parameters

param = {"C" : [1,10,100], "gamma": [1,.01,.001], "kernel" : ['linear']}

grd_search = GridSearchCV(SVC(),param,verbose=3)
#fitting the parameter 

#grd_search.fit(X_train,y_train)
#grd_search.best_params_#
#predicting the Test data

#pred = grd_search.predict(X_test)
#print (classification_report(pred,y_test))
#print(confusion_matrix(pred,y_test))