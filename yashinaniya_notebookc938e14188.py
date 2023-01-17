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
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
dataset = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
combine = [dataset, test_data]
dataset.head()
ax = sns.countplot(x='Sex', hue='Survived', data=dataset)
#plt.show()

col = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
no_of_rows = 2
no_of_col = 3
fig, axs = plt.subplots(no_of_rows, no_of_col, figsize=(no_of_col * 3.5, no_of_rows * 3))

for r in range(0, no_of_rows):
    for c in range(0, no_of_col):
        i = r * no_of_col + c
        ax = axs[r][c]
        sns.countplot(dataset[col[i]], hue=dataset["Survived"], ax=ax)
        ax.set_title(col[i], fontsize=14, fontweight='bold')
        ax.legend(title="survived", loc='upper center')

plt.tight_layout()
#use plt.show() if you are using an IDE like Pycharm
g = sns.FacetGrid(dataset, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(dataset, col='Pclass', hue='Survived')
grid = sns.FacetGrid(dataset, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
for dat in combine:
    dat['Title'] = dat.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dat in combine:
    dat['Title'] = dat['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dat['Title'] = dat['Title'].replace('Mlle', 'Miss')
    dat['Title'] = dat['Title'].replace('Ms', 'Miss')
    dat['Title'] = dat['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dat in combine:
    dat['Title'] = dat['Title'].map(title_mapping)
    dat['Title'] = dat['Title'].fillna(0)

dataset = dataset.drop(['Name', 'PassengerId', 'Cabin', 'Embarked', 'Ticket','Fare'], axis=1)

test_data = test_data.drop(['Name', 'PassengerId', 'Cabin', 'Embarked', 'Ticket', 'Fare'], axis=1)
combine = [dataset, test_data]
dataset.head()

test_data.head()
X = dataset.iloc[:, 1:].values
X_test = test_data.iloc[:, :].values
y = dataset.iloc[:, 0].values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])
imputer.fit(X_test[:, 2:3])
X_test[:, 2:3] = imputer.transform(X_test[:, 2:3])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# Encoding the test set
c_t = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_test = np.array(c_t.fit_transform(X_test))
# print(X)
# Training the SVM model on the Training set
from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X, y)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred.reshape(len(y_pred), 1))