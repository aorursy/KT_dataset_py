# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import sklearn as skl

from sklearn.model_selection import train_test_split

from sklearn import cross_validation

# presentation

import seaborn as sns

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.



print(sklearn.__version__)
# load data

df = pd.read_csv("../input/Iris.csv")

df.shape
df.describe()
df.head()
# drop irrelavent columns

df.drop(["Id"], axis=1, inplace=True)
df["Species"].value_counts()
sns.jointplot(x="SepalLengthCm",y="SepalWidthCm",data=df, size=8)
# raw species distribution

sns.FacetGrid(df, hue="Species", size=6).map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend()
# plot individual feature in seaborn by a boxplot

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(10,5))

sns.boxplot( x='Species', y="SepalLengthCm", data=df, ax=ax1)

sns.boxplot( x='Species', y="SepalWidthCm", data=df, ax=ax2)

sns.boxplot( x='Species', y="PetalLengthCm", data=df, ax=ax3)

sns.boxplot( x='Species', y="PetalWidthCm", data=df, ax=ax4)
# We will split the loaded dataset into two, 

# 80% of which we will use to train our models 

# and 20% that we will hold back as a validation dataset.



array = df.values

X = array[:,0:4]

Y = array[:,4]

test_portion=0.2

seed=7



X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=test_portion,random_state=seed)



# Test options and evaluation metric

num_folds = 10

num_instances = len(X_train)

seed = 7

metric = 'accuracy'



#Let’s evaluate 6 different algorithms:

# -- Logistic Regression (LR)

# -- Linear Discriminant Analysis (LDA)

# -- K-Nearest Neighbors (KNN).

# -- Classification and Regression Trees (CART).

# -- Gaussian Naive Bayes (NB).

# -- Support Vector Machines (SVM).



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))



# model iteration

results=[]

names=[]

for name, model in models:

    cv = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

    cv_results = cross_validation.cross_val_score(model,X_train,Y_train,cv=cv, scoring=metric)

    results.append(cv_results)

    names.append(name)

    print(name, cv_results.mean(), cv_results.std())
# Use SVM to perform the final prediction

svm = SVC()

svm.fit(X_train,Y_train)

pred = svm.predict(X_test)

svm.score(X_test,Y_test)