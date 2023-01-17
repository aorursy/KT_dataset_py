# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



test_full = pd.read_csv("../input/titanic/test.csv",index_col="PassengerId")

train_full = pd.read_csv("../input/titanic/train.csv",index_col = "PassengerId")



target = train_full.Survived

train = train_full.drop(['Name','Survived'],axis=1)

test = test_full.drop(['Name'],axis=1)

sns.pairplot(train_full)

plt.figure(figsize=(12,10))

sns.heatmap(train_full.corr(),annot=True)
[(col,train[col].nunique()) for col in train.select_dtypes('object')]
train.isnull().sum()
train
test
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent',add_indicator=True)



train_preprocessed = pd.DataFrame(imputer.fit_transform(train),columns=[col for col in train] + ['Age_na','Cabin_na','Embarked_na'],index = train.index)

test_preprocessed =  pd.DataFrame(imputer.transform(test),columns=[col for col in test] + ['Age_na','Cabin_na','Embarked_na'],index = test.index)

train_preprocessed
from sklearn.preprocessing import LabelEncoder



LE = LabelEncoder()



for col in [col for col in train.select_dtypes('object')]:

    train_preprocessed[col] = LE.fit_transform(train_preprocessed[col])

    test_preprocessed[col] = LE.fit_transform(test_preprocessed[col])



train_preprocessed = train_preprocessed.astype('float64')

test_preprocessed = test_preprocessed.astype('float64')

train_preprocessed
X_train,X_test,y_train,y_test = train_test_split(train_preprocessed,target,test_size=0.2)

display(X_train)
from xgboost import XGBClassifier

from sklearn.metrics import mean_absolute_error,accuracy_score,f1_score



model = XGBClassifier(n_estimators=90,n_jobs=5,random_state=0)



model.fit(X_train,y_train)

preds = model.predict(X_test)



'Accuracy Score is %f' % accuracy_score(y_pred=preds,y_true=y_test)
mean_absolute_error(preds,y_test)
f1_score(preds,y_test)
submission_preds = model.predict(test_preprocessed)

output = pd.read_csv("../input/titanic/gender_submission.csv")

output.Survived = submission_preds

output.to_csv('submission.csv',index=False)

output.head()