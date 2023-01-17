# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
## accuracy score before using GridSearchCV

from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
pm = {

    'criterion':['gini', 'entropy'],

    'max_depth':[1,2,3,4,5,6,7,None]

}
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(clf,param_grid=pm, cv=10, n_jobs=-1)
grid.fit(X_train,y_train)
## accuracy score after using GridSearchCV

grid.best_score_
## Best parameteres...

grid.best_params_