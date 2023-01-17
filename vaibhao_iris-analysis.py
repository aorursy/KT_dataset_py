# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/Iris.csv" )
df.head()
sns.pairplot(df,palette='husl',hue = 'Species')
#using Svm to figure out the Species

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
#importing the gridsearch

from sklearn.grid_search import GridSearchCV
#spliting traing and testing data

X = df.drop('Species',axis = 1)

y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#defining parameter for grid search

param = {'C': [0.1,1,10,100,1000], "gamma":[1,0.1,0.01,0.001, 0.0001],'kernel' : ['linear', 'rbf','sigmoid']}

grd_search = GridSearchCV(SVC(),param, verbose=2)
#fitting the Traing data on grid search model

grd_search.fit(X_train,y_train)
#Getting the best parameter

grd_search.best_params_
#prediction

pred = grd_search.predict(X_test)
#grd_search.best_estimator_
df.columns
a = {"Id":999,'SepalLengthCm':3.5,'SepalWidthCm':3.8,'PetalLengthCm':4.5,'PetalWidthCm':.6}

pd.Series(a)

X_test.append(a,ignore_index=True)
#grd_search.predict(newly added data)

grd_search.predict(X_test[-1:])

len(X_test)
print(classification_report(pred,y_test))
print(confusion_matrix(pred,y_test))
svm = SVC(C= 1,gamma = 1, kernel = 'linear')
svm.fit(X_train,y_train)
p = svm.predict(X_test)
print (classification_report(p,y_test))
print (confusion_matrix(p,y_test))