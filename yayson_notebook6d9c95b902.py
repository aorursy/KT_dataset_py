# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



titanic_df.head()
titanic_df.info()

print("------------------------")

test_df.info()
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

test_df = test_df.drop(['Name', 'Ticket'], axis=1)


titanic_df.info()

print("--------------------------")

test_df.info()
# fill na

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")



#plot

sns.factorplot('Embarked', 'Survived', data=titanic_df, size=3, aspect = 2)



fig, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(8,3))



sns.countplot(x='Embarked', data=titanic_df, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=titanic_df, ax=axis2)



embark_pct = titanic_df[["Embarked", "Survived"]].groupby(["Embarked"], as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_pct, order=['S','C','Q'], ax=axis3)

titanic_df.info()




embark_dummies_titanic  = pd.get_dummies(titanic_df["Embarked"])

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(test_df["Embarked"])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



titanic_df = titanic_df.join(embark_dummies_titanic)

test_df    = test_df.join(embark_dummies_test)



titanic_df.drop(["Embarked"], axis=1,inplace=True)

test_df.drop(["Embarked"], axis=1,inplace=True)
titanic_df.info()



titanic_df.head()
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



# convert from float to int

titanic_df['Fare'] = titanic_df['Fare'].astype(int)

test_df['Fare']    = test_df['Fare'].astype(int)



# get fare for survived & didn't survive passengers 

fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]

fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]



# get average and std for fare of survived/not survived passengers

avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])



# plot

titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)