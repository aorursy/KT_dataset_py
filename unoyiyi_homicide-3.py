# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline

!pip install ggplot

from ggplot import *



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#load dataset

database_df=pd.read_csv('../input/database.csv')

database_df.head()
database_df.info()
database_df["Perpetrator Age"]= pd.to_numeric(database_df["Perpetrator Age"], errors='coerce').fillna(0).astype(int)
database_df.info()
def minMax(x):

    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

database_age = database_df[["Record ID","Victim Age","Perpetrator Age","Victim Count","Perpetrator Count"]]

database_age.apply(minMax)
#drop Victim Age>99 & Perpetrator Age >99 & Victim Age and Perpetrator Age = 0

database_df1 = database_df.drop(database_df[(database_df["Victim Age"]> 98) | (database_df["Perpetrator Age"]>98)|(database_df["Perpetrator Age"] < 1)|(database_df["Victim Age"]< 1)].index)

##The operators are: | for or, & for and, and ~ for not. These must be grouped by using parentheses.
def minMax(x):

    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

database_age = database_df1[["Record ID","Victim Age","Perpetrator Age","Victim Count","Perpetrator Count"]]

database_age.apply(minMax)
#1

p = ggplot(database_df1, aes(x="Victim Age", y = "Perpetrator Age")) + geom_point(size = 10, color = 'blue')+stat_smooth(color = 'red', se=False, span=0.2) + facet_grid('Crime Solved')

p + xlab("Victim Age") + ylab("Perpetrator Age") + ggtitle("Victim Age vs Perpetrator Age on whether it was solved") 
#2

#seaborn heatmap

plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("Correlation Heatmap")

corr = database_df.corr()

sns.heatmap(corr, annot=True,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
#3

from sklearn.model_selection import train_test_split

X, y = database_df.iloc[:, 1:].values, database_df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=56)
X_train