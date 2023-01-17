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
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df= pd.read_csv('/kaggle/input/hr-analytics/HR_comma_sep.csv')
df.head()
df.groupby('left').mean()
pd.crosstab(df.salary,df.left).plot(kind='bar')
pd.crosstab(df.Department,df.left).plot(kind='bar')
df1=df[['satisfaction_level','promotion_last_5years','average_montly_hours','salary']]
df1.head()
sal_dummies= pd.get_dummies(df1['salary'],prefix="salary")
df_with_dummies = pd.concat([df1,sal_dummies],axis='columns')

df_with_dummies.drop('salary',axis=1, inplace=True)
df_with_dummies.head()
X=df_with_dummies
y=df['left']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=46)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred= model.predict(X_test)
model.score(X_test,y_test)
from sklearn.model_selection import GridSearchCV
params={"C":np.linspace(.01, 100, 10)}
model=LogisticRegression(max_iter=15000)
logreg_cv=GridSearchCV(model,params,cv=10,scoring= 'accuracy')
logreg_cv.fit(X_train,y_train)
logreg_cv.best_params_
lr=LogisticRegression(C=0.01, penalty= 'l2')
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
# import the library 
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV