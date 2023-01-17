# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('../input/digit-recognizer/train.csv')

test=pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
print(train.shape)

print(test.shape)
X=train.iloc[:,1:].values
y=train.iloc[:,0].values
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=9)
X_train.shape
from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_test[100]
plt.imshow(X_test[100].reshape(28,28))

clf.predict(X_test[100].reshape(1,784))
y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
param_dist={

    "criterion":["gini","entropy"],

    "max_depth":[1,2,3,4,5,6,7,8,9,10,None],

    "splitter" : ["best", "random"]

}
from sklearn.model_selection import GridSearchCV #using GridSearchCV for better accuracy

grid=GridSearchCV(clf, param_grid=param_dist, cv=10, n_jobs=-1)
grid.fit(X_train, y_train)
grid.best_estimator_
grid.best_score_
X_final=test.iloc[:,:].values
y_final=grid.predict(X_final)
results = pd.Series(y_final,name="Label")

new = pd.concat([pd.Series(range(1,y_final.shape[0]+1),name = "ImageId"),results],axis=1)

new

new.to_csv('submission.csv',index=False)
