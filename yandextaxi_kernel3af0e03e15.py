# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



test = pd.read_csv("/kaggle/input/titanic/test.csv")

test_shape = test.shape

print(test_shape)
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train_shape = train.shape

print(train_shape)
train.head(10)
import matplotlib.pyplot as plt



sex_pivot = train.pivot_table(index="Sex",values="Survived")

sex_pivot

pclass_pivot = train.pivot_table(index="Pclass",values="Survived")

pclass_pivot.plot.bar()

plt.show()
train['Age'].describe()
train[train["Survived"] == 1]
survived = train[train["Survived"] == 1]

died = train[train["Survived"] == 0]

survived["Age"].plot.hist(alpha=0.5,color='yellow',bins=50)

died["Age"].plot.hist(alpha=0.5,color='black',bins=50)

plt.legend(['Survived','Died'])

plt.show()
def process_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df



cut_points = [-1,0,18,100]

label_names = ["Missing","Child","Adult"]



train = process_age(train,cut_points,label_names)

test = process_age(test,cut_points,label_names)
def process_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df



cut_points = [-1,0, 5, 12, 18, 35, 60, 100]

label_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']



train = process_age(train,cut_points,label_names)

test = process_age(test,cut_points,label_names)



age_cat_pivot = train.pivot_table(index="Age_categories",values="Survived")

age_cat_pivot.plot.bar()

plt.show()
train['Pclass'].value_counts()
column_name = "Pclass"

df = train

dummies = pd.get_dummies(df[column_name],prefix=column_name)

dummies.head()
def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



train = create_dummies(train,"Pclass")

test = create_dummies(test,"Pclass")

train.head()
train = create_dummies(train,"Sex")

test = create_dummies(test,"Sex")

train = create_dummies(train,"Age_categories")

test = create_dummies(test,"Age_categories")
train.head()
holdout = test # from now on we will refer to this

               # dataframe as the holdout data



from sklearn.model_selection import train_test_split



columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']



all_X = train[columns]

all_y = train['Survived']



train_X, test_X, train_y, test_y = train_test_split(

    all_X, all_y, test_size=0.2,random_state=0)
train_X.shape
from sklearn.metrics import accuracy_score
holdout.head()