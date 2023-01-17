# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import seaborn as sb

import matplotlib.pyplot as plt

import sklearn



from pandas import Series, DataFrame

from pylab import rcParams

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn import metrics 

from sklearn.metrics import classification_report

from patsy import dmatrices



# Any results you write to the current directory are saved as output.
dftest = pd.read_csv("../input/test.csv")

dftest.head()
df = pd.read_csv("../input/train.csv")

df.head()

df.Fare.describe()
df.Sex.unique()
df['AgeIndex'] = df['Age'].apply(lambda x: 1 if x <= 15 else 2 if x <= 40 else 3)

dftest['AgeIndex'] = dftest['Age'].apply(lambda x: 1 if x <= 15 else 2 if x <= 40 else 3)



df.head()
df['SexIndex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 2)

dftest['SexIndex'] = dftest['Sex'].apply(lambda x: 1 if x == 'male' else 2)



df.head()
dffinal = df[['Survived','Pclass','SibSp','Parch','Age','Fare','SexIndex','AgeIndex']]

dffinaltest = dftest[['Pclass','SibSp','Parch','Age','Fare','SexIndex','AgeIndex']]



dffinal.head()
dffinaltest.head()
pd.isnull(dffinal).sum()
pd.isnull(dffinaltest).sum()
dffinal.Fare.describe()
values = {'Age': 29}

dffinal.fillna(29, inplace=True)

dffinaltest.fillna(29, inplace=True)
y, X = dmatrices('Survived ~ Pclass + SibSp + Parch + Age + Fare + SexIndex + AgeIndex', dffinal,  return_type="dataframe")

X.columns
y.head()
y = np.ravel(y)

#y
# instantiate a logistic regression model, and fit with X and y

model = LogisticRegression()

model = model.fit(X, y)



# check the accuracy on the training set

model.score(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model2 = LogisticRegression()

model2.fit(X_train, y_train)
predicted = model2.predict(X_test)

predicted[0:10]
X_train.head()
dffinaltest['intercept'] = 1

finalpredicted = model2.predict(dffinaltest)

finalpredicted[0:10]
probs = model2.predict_proba(X_test)

print (probs)
print (metrics.accuracy_score(y_test, predicted))

print (metrics.roc_auc_score(y_test, probs[:, 1]))
zip(X.columns, np.transpose(model.coef_))
dftest.shape
sol = pd.DataFrame(finalpredicted)

sol.columns = {'Survived'}

sol.head()
result = pd.concat([dftest, sol], axis=1, join="inner")

result.head()
result[['PassengerId','Survived']]
result[['PassengerId','Survived']].to_csv('output.csv', index = False)