# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/titanic_data.csv')

df.shape



# 891 rows, 12 features
df.head()
df['Age'].values
# to see data in row 55

df.values[55]
# make column name to array of column names

df[['Name', 'Survived','Age']].values[55:60]
# df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

df.head()

# df = df.dropna()

df.shape



sexConvDict = {'male':1, 'female':2}

df['Sex'] = df['Sex'].apply(sexConvDict.get).astype(int)

df.head()
# decission tree

from sklearn import tree

from sklearn.model_selection import train_test_split



features = ['Sex']

X = df[features].values

y = df['Survived'].values



# native array python

# feature_train, feature_test, label_train, label_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

#test

y_predicted = clf.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_predicted)



# ( [TN, FP],

#   [FN, TP])
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predicted)
# now we know that sex is one feature that significant
# 0.77

features = ['Sex']



# 0.61

features = ['Parch']



#0.66

features = ['Pclass']



#0.58

features = ['Age']



#0.65

features = ['Fare']



#0.57

features = ['SibSp']



# 0.78

features = ['Sex', 'Pclass']



# 0.80

features = ['Sex', 'Pclass', 'Age']



X = df[features].values

y = df['Survived'].values



# native array python

# feature_train, feature_test, label_train, label_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)

accuracy_score(y_test, y_predicted)

#neuron network

from sklearn.neural_network import MLPClassifier



# 0.77

features = ['Sex']



# 0.61

features = ['Parch']



#0.66

features = ['Pclass']



#0.58

features = ['Age']



#0.65

features = ['Fare']



#0.57

features = ['SibSp']



# 0.78

features = ['Sex', 'Pclass']



# 0.80

features = ['Sex', 'Pclass', 'Age']

X = df[features].values

y = df['Survived'].values



# pre-processing data for neuron network



scaler = StandardScaler()

X_standard = scaler.fit_transform(df[features].values)



# native array python

# feature_train, feature_test, label_train, label_test

X_train, X_test, X_std_train, X_std_test, y_train, y_test = train_test_split(X, X_standard, y, test_size=0.50, random_state=1)

clf = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(10,15), random_state=0)

# clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)

accuracy_score(y_test, y_predicted)



# non-std 0.60784313725490191

# std 0.79271708683473385