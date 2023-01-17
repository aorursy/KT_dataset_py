# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection  import train_test_split, GridSearchCV

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.tree import  DecisionTreeClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
columns=['buying','maint','doors','persons','lug_boot','safety','decision']

df=pd.read_csv('/kaggle/input/car-evaluation-data-set/car_evaluation.csv')
df.columns=columns
df.head(10)
df.columns
df.info()
df['decision'].unique()
df.describe()
X=df.drop('decision',axis=1)

y=df['decision']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)
import category_encoders as ce



encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])



X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
X_train.head()
X_test.head()
params={

    'criterion':['gini','entropy'],

    'class_weight':[None,'balanced']

}
clf=DecisionTreeClassifier()

clf_search=GridSearchCV(clf,params,cv=5)

clf_search.fit(X_train,y_train)
clf_search.best_params_
clf_gini = DecisionTreeClassifier(criterion='entropy',class_weight=None, random_state=0)





# fit the model

clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)
from sklearn.metrics import accuracy_score



print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

y_pred_test_gini = clf_gini.predict(X_test)

acc=accuracy_score(y_test,y_pred_test_gini)

acc