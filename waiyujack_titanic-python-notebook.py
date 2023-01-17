# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))  

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
#load the data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_all = [df_train, df_test]
#show first(or last) 5 rows of the train data
df_train.head() #train_df.tail()
df_train.describe()
df_all[:891].info()
print('*'*40)
df_all[891:].info()
## sex pivot
sex_pivot=df_train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()
# class pivot
class_pivot = df_train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
## fare and sex pivot
fare_cut_train=pd.cut(df_train["Fare"],[0,5,10,25,50,75,100,10000])
df_train["fare_cut"]=fare_cut_train
fare_cut_test=pd.cut(df_test["Fare"],[0,5,10,25,50,75,100,10000])
df_test["fare_cut"]=fare_cut_test
df_train.pivot_table("Survived",index="fare_cut",columns='Sex',aggfunc='mean')
df_train.pivot_table("Survived",index="fare_cut",columns='Sex',aggfunc='count')

## Age investigation
df_train["Age"].describe()
survived = df_train[df_train["Survived"] == 1]
died = df_train[df_train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()

## create age categories variable based on age
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

df_train = process_age(df_train,cut_points,label_names)
df_test = process_age(df_test,cut_points,label_names)
## show survival rate by age categories
pivot = df_train.pivot_table(index="Age_categories",values='Survived')
pivot.plot.bar()
plt.show()
df_train.pivot_table(index="Parch",values="Survived")
df_train.pivot_table(index="Parch",columns='Sex',values="Survived")
df_train.pivot_table(index="Parch",columns='Age_categories',values="Survived")
df_train.pivot_table(index="Age_categories",columns='Parch',values="Survived")
df_train.pivot_table(index="Age_categories",columns='Parch',values="Survived",aggfunc='count')
df_train.pivot_table(index="Parch",values="Survived",aggfunc="count")
df_train.pivot_table(index="SibSp",values="Survived")
df_train.pivot_table(index="SibSp",values="Survived",aggfunc="count")