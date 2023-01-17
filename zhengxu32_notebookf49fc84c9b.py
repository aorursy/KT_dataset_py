# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn import model_selection

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_comma_sep.csv')
df.columns
df.head()
df.describe()
df[df['left'] == 1].describe()
df[df['left'] == 0].describe()
df['salary'] = df['salary'].map({'low':0,'medium':1,'high':2})
df.columns
train = df.drop(['left', 'sales'], 1)

test = df['left']
X_train, X_test, y_train, y_test = model_selection.train_test_split(train, test, test_size=0.3)
clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train)



accuracy = clf.score(X_test, y_test)

print(accuracy)
clf2 = RandomForestClassifier(random_state=0)

clf2.fit(X_train, y_train)



accuracy2 = clf2.score(X_test, y_test)

print(accuracy2)