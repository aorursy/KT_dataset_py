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
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
%matplotlib inline
sns.set()
# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])
data.info(
)
# Impute missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()
df_train.shape
df_test.shape

df_test.head()
df_train.head()
print(df_train[df_train.Sex == 'female'].Name.sum())
print(df_train[df_train.Sex == 'female'].Survived.count())

print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'female'].Survived.value_counts())
data = pd.get_dummies(data, columns=['Sex'],drop_first=True)
data.head(
)
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()
example_answer = pd.read_csv('../input/gender_submission.csv')
example_answer.shape
data.info()
data_train = data.iloc[:891]
data_test = data.iloc[891:]
data_train.head()
data_test.head()
data_test.shape
data_train.shape
#transformng data from Data frame to Array 

X = data_train.values
test = data_test.values
y = survived_train.values
# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred
my_prediction = df_test[['PassengerId', 'Survived']]
my_prediction.head()
my_prediction.to_csv('my_prediction.csv', index = False)
