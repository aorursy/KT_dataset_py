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
# Imports



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
titanic_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



titanic_df.head(10)
titanic_df.info()

print("------------------")

test_df.info()

titanic_df.drop(["PassengerId","Name","Ticket"],axis=1)

test_df.drop(["PassengerId","Name","Ticket"],axis=1)
titanic_df.describe()
sns.jointplot(x='Survived',y='Pclass',data=titanic_df[['Survived','Pclass']])
with sns.axes_style("white"):

    sns.jointplot(x='Survived',y='Pclass',data=titanic_df[['Survived','Pclass']], kind="hex", color="k");
with sns.axes_style("white"):

    sns.jointplot(x='Age',y='Fare',data=titanic_df[['Age','Fare']], kind="hex", color="k");
with sns.axes_style("white"):

    sns.jointplot(x='Age',y='Fare',data=titanic_df[['Age','Fare']]);
sns.pairplot(titanic_df)
sns.pairplot(titanic_df[['Fare','Pclass','Survived']])