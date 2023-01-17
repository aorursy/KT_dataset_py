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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

%matplotlib inline





data = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

data.head(10)
data.describe()
sns.heatmap(data.corr(),cmap='Greens')
data.corrwith(data['DEATH_EVENT'])
features = data[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]

sns.pairplot(features)
X = features.values

y = data['DEATH_EVENT'].values



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
lr = LogisticRegression()

lr.fit(X_train,y_train)

predictions_LR = lr.predict(X_test)

acc_LR = cross_val_score(lr,X_train,y_train,cv=10)

score = lr.score(X_test,y_test)

print('LogisticRegression score:  ',score)
knn = KNeighborsClassifier(n_neighbors=12,p=2)

knn.fit(X_train,y_train)

predictions_KNN = knn.predict(X_test)

acc_KNN = cross_val_score(knn,X_train,y_train,cv=10)

score = knn.score(X_test,y_test)

print('KNeighborsClassifier score:  ',score)
dtc = DecisionTreeClassifier(max_depth=2,max_features=3,)

dtc.fit(X_train,y_train)

predictions_DTC = dtc.predict(X_test)

acc_DTC = cross_val_score(dtc,X_train,y_train,cv=10)

score = dtc.score(X_test,y_test)

print('DecisionTreeClassifier score:  ',score)
rfc = RandomForestClassifier(n_estimators=200,criterion='entropy',max_depth=4)

rfc.fit(X_train,y_train)

predictions_RFC = rfc.predict(X_test)

acc_RFC = cross_val_score(rfc,X_train,y_train,cv=10)

score = rfc.score(X_test,y_test)

print('RandomForestClassifier score:  ',score)
gbc = GradientBoostingClassifier(max_depth=2)

gbc.fit(X_train,y_train)

predictions_GBC = gbc.predict(X_test)

acc_GBC = cross_val_score(gbc,X_train,y_train,cv=10)

score = gbc.score(X_test,y_test)

print('GradientBoostingClassifier score:  ',score)