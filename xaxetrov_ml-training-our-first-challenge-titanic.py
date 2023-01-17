import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split


print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# Load test & train data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# Lets understand  the csv files

train.head()
train.describe(include=np.number)
train.describe(include=np.object)
X = pd.get_dummies(train[['Fare', 'Sex']])

y = train['Survived']



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



X.describe()
# y histogram

sns.distplot(y, kde = False)

plt.show()
# Fare histogram -> check split

fig, ax = plt.subplots()

sns.distplot(X['Fare'], kde = False, ax=ax) # sns.distplot(X['Fare'], hist = False, ax=ax)

sns.distplot(train_X['Fare'], kde = False, ax=ax) # sns.distplot(train_X['Fare'], hist = False, ax=ax)

plt.show()
#skewness and kurtosis

print("Skewness: %f" % X['Fare'].skew())  #  third standardized moment

print("Kurtosis: %f" % X['Fare'].kurt())  #  fourth standardized moment
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True, annot=True);
# Relation between class and survival

sns.barplot(x='Pclass', y='Survived', data=train, ci=None)

plt.show()
model = LogisticRegression(random_state=1, multi_class='auto', solver='lbfgs')

model.fit(train_X, train_y)



pred_y = model.predict(val_X)



print (accuracy_score(pred_y, val_y))