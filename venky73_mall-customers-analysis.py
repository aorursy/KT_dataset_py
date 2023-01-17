# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Mall_Customers.csv")
df.columns
df.info()
df.describe()
df.columns = ["ID","Gender" ,"Age","Income","Score"]
plt.hist(df.Age)
df.boxplot()
df["Income"]
df.head()
df["Gender"].value_counts()
df["Gender"] = df.Gender.map({"Male":1,"Female":0})
sns.barplot(x = "Gender", y = "Score", data = df)
sorted(list(df.Age.unique()))
def group_age(age):

    if age <= 30:

        return "<= 30"

    elif age <= 40:

        return "31-40"

    elif age<= 50:

        return "41-50"

    elif age<= 60:

        return "51-60"

    else:

        return "61-70"
df["Age"]= df.Age.apply(group_age)
df
sns.barplot(x = 'Age', y = "Score", hue = "Gender", data  = df,)
df.Age.unique()
sns.barplot(x = "Age",y="Income",hue = "Gender", data = df, order = ['<= 30', '31-40', '41-50','51-60','61-70'])
plt.scatter(x = "Income", y = "Score", data = df)
plt.scatter(x = "Income", y = "Score", data = df[df.Gender == 1])
plt.scatter(x = "Income", y = "Score", data = df[df.Gender == 0])
plt.scatter(x = "Age", y = "Score", data = df[df.Gender == 0])
plt.scatter(x = "Age", y = "Score", data = df[df.Gender == 1])
df.groupby(["Gender","Age"]).mean()