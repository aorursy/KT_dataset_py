# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Plotting

# Input data files are available in the "../input/" directory.

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/train.csv') #Loads train.csv into a Pandas Dataframe
# df.head() to see first 5 rows of table

df.head()
# df.shape to see shape of table (how many rows and columns)

df.shape
X = df.Age

y = df.Survived



plt.scatter(X, y)
# Querying in Pandas

children = df[(df.Age <= 18)]

children.shape
children.Survived.value_counts()
adults = df[(df.Age > 18)]

adults.Survived.value_counts()
females = df[(df.Sex == 'female')]

females.shape
females.Survived.value_counts()
males = df[(df.Sex == 'male')]

males.Survived.value_counts()
109 / (109 + 468)
df.shape
342 / 891
gender_map = {"male": 0, "female": 1}

df['Sex'] = df.Sex.map(gender_map)
#initialize the model

clf = RandomForestClassifier(n_estimators=1500)

#fit data into the model

y = df.Survived.values

X = df[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare']]

clf.fit(X, y)
df = df[(~df.Age.isnull())]
scores = cross_val_score(clf, X, y)
scores.mean()
clf.feature_importances_