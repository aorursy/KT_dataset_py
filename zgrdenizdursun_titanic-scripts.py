# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')

test_df=pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()
def bar_plot(variable):

    var=train_df[variable]

    varValue=var.value_counts()

    #visualize 

    plt.figure(figsize=(9,3)) #olmayadabilir. 

    plt.bar(varValue.index,varValue)

    #plt.xticks

    plt.xticks(varValue.index, varValue.index.values) 

    plt.ylable="Frequency"

    plt.title(variable)

    plt.show()

    print("{} : \n {}".format(variable,varValue))

    

    
category1 = ["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:

    bar_plot(c)
def hist_plot(variable):

    plt.figure(figsize=(9,3))

    plt.hist(train_df[variable],bins=50,color='r')

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distrubition with hist".format(variable))

    plt.show()

    
numericVar=["Fare","Age","PassengerId"]

for n in numericVar:

    hist_plot(n)
train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean()

train_df[["Pclass","Age"]].groupby(["Pclass"],as_index=False).mean()
train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean()
train_df[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean().sort_values(by="Survived",ascending=False)
train_df[["Embarked","Fare"]].groupby(["Embarked"],as_index=False).mean().sort_values(by="Fare",ascending = False)
train_df[["Parch","Survived"]].groupby(["Parch"],as_index=False).mean().sort_values(by="Survived",ascending=False)
train_df[train_df["Embarked"].isnull()]

train_df["Embarked"]=train_df["Embarked"].fillna("C")
train_df.columns
