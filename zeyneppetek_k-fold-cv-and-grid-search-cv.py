# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/insurance/insurance.csv")

data.info()

data.head()

y=data["sex"]

x=data.iloc[:,[2,3]].values



data.head()

#normalization

x=(x-np.min(x))/(np.max(x)-np.min(x))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3)
#KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=knn, X=x_train, y=y_train,cv=10)

#estimator knn seçilir çünkü datayı knn kadar parçaya ayırıp hepsini tahmin edecek

print("average accuracy:",np.mean(accuracies))

print("stdeviation accuracy:",np.std(accuracies))
knn.fit(x_train,y_train)

print("test accuracy:",knn.score(x_test,y_test))
#Grid Search CV

from sklearn.model_selection import GridSearchCV

grid={"n_neighbors":np.arange(1,50)}

knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn,grid, cv=10)

knn_cv.fit(x_train,y_train)

# en iyi knn değerini grid searchle buluruz.



print("tuned hyperparameter K:",knn_cv.best_params_) #en iyi parametreyi seçiyoruz.

print("best score:",knn_cv.best_score_) 
#Grid search cv with Logistic Reg



from sklearn.linear_model import LogisticRegression

param_grid={"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}



logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,grid,cv=10)

logreg_cv.fit(x_train,y_train)

print("tuned hyperparameters:(best parameters):",logreg_cv.best_params_)

print("accuracy:",logreg_cv.best_score_)


logreg2=LogisticRegression(C=1,penalty="l1")

logreg2.fit(x_train,y_train)

print("score:",logreg2.score(x_test,y_test))
