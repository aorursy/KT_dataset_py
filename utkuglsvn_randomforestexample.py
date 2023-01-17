# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')

data.head()

#Bu veri setinde müşterilerin önceki davranışlara göre aylık parayı ödeyip ödemediklerini analiz edeceğiz
data.info()
import matplotlib.pyplot as plt

data.hist('SEX')
data.hist('AGE')
data.hist('EDUCATION')
from sklearn.model_selection import train_test_split

X=data.drop('default.payment.next.month',axis=1)

y=data['default.payment.next.month']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc
model=rfc.fit(X_train,y_train)

model
predictions=model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

print("accuracy:",accuracy_score(y_test,predictions))

print("accuracy:",precision_score(y_test,predictions))

print("accuracy:",recall_score(y_test,predictions))

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, predictions))
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



n_folds=5

paramaters={'max_depth': range(2,20,5)}

rfc=RandomForestClassifier()



rfc=GridSearchCV(rfc,paramaters,cv=n_folds,scoring="accuracy",return_train_score=True)

rfc.fit(X_train,y_train)
scores=rfc.cv_results_

pd.DataFrame(scores).head()
import matplotlib.pyplot as plt

plt.figure()

plt.plot(scores["param_max_depth"],scores["mean_train_score"],label="training accuracy")

plt.plot(scores["param_max_depth"],scores["mean_test_score"],label="test_accuracy")

plt.xlabel("max_depth")

plt.ylabel("accuracy")

plt.legend()

plt.show()
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



n_folds=5



parameters={'n_estimators': range(100,700,400)}

rf=RandomForestClassifier(max_depth=4)

rf=GridSearchCV(rf,parameters,cv=n_folds,scoring="accuracy",return_train_score=True)

rf.fit(X_train,y_train)

scores=rf.cv_results_

pd.DataFrame(scores).head()
plt.figure()

plt.plot(scores["param_n_estimators"],scores["mean_train_score"],label="training accuracy")

plt.plot(scores["param_n_estimators"],scores["mean_test_score"],label="test_accuracy")

plt.xlabel("n estimatros")

plt.ylabel("accuracy")

plt.legend()

plt.show()
from sklearn.ensemble import RandomForestRegressor

param_grid = {

    'bootstrap': [True],

    'max_depth': [4, 8, 12],

    'max_features': [2, 4,8],

    'min_samples_leaf': [3, 5, 7],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [50, 100, 200]

}

rf = RandomForestRegressor()

rfc = RandomForestRegressor()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search2 = GridSearchCV(estimator = rfc, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train,y_train)
#en iyi parametre

print('Best paramters:',grid_search.best_params_)

pd.DataFrame(grid_search.cv_results_).head()
grid_search2.fit(X_train,y_train)
#en iyi parametre

print('Best paramters:',grid_search2.best_params_)
scores=grid_search2.cv_results_

pd.DataFrame(scores).head()
rfc=RandomForestClassifier(bootstrap=True,max_depth=12, max_features= 4, min_samples_split= 7,min_samples_leaf=12, n_estimators=200)

rfc_tuned=rfc.fit(X_train,y_train)

rfc_tuned
y_pred=rfc_tuned.predict(X_test)

accuracy_score(y_test,y_pred) #acc skor
print(classification_report(y_test,y_pred))