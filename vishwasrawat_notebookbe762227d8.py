# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#sample submission file
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
cols = df_train.columns
print(cols)
print(df_train.shape)
df.head(10)
df_train.info()
print(cols)
for col in cols:
    print(col + "\t" + str(df_train[col].isnull().sum()))
for col_1 in cols: 
    print(col_1)
    print(df_train[col_1].value_counts())
num_cols = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
for col_1 in num_cols: 
    print(col_1)
    print("Min: " + str(df_train[col_1].min()) + "\tMax: " + str(df_train[col_1].max()))
#categorical to numerical: 
df_1 = df_train.drop(columns = ["PassengerId", "Name", "Cabin", "Ticket"])
embarked_dict = {"S": 2, "C": 1, "Q": 0}
sex_dict = {"male": 1, "female": 0}
df_1 = df_1.replace({"Embarked" : embarked_dict, "Sex": sex_dict})
df_1.head(10)
import matplotlib.pyplot as plt
plt.figure(figsize=(14,10))
sb.heatmap(df_1.corr(), annot = True)
import seaborn as sb
sb.distplot(df_train["Age"])

#some unusialty observed here is the number of poeple below age 10 in the dataset - highly unlikely but can be. 
#but the number of people below age 1- need to be treated - these can be mistyped values or might seriously be age. 
df_train["Age"][df_train["Age"]<1]
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_train.head(10)
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_test.head(10)
print(df_train.info())

print(df_test.info())
df_train.loc[df_train["Age"].isnull()]
df_train.loc[df_train["Embarked"].isnull()]
df1 = df_train.copy()
df1 = df1.drop(columns = ["Cabin"])
df1 = df1.dropna()
df1.info()
df1.corr()
import seaborn as sb 
sb.heatmap(df1.corr(), annot=True)
df_train["Cabin"].value_counts()
df1.head()
df1["Sex"].value_counts()
df1["Sex"] = df1["Sex"].apply(lambda x: 1 if(x=="male") else 0)
df1.head()
df1["Embarked"].value_counts()
embarked_dict = {"S": 0, "C": 1, "Q": 2}
df1 = df1.replace({"Embarked" : embarked_dict})
df1.head()
sb.heatmap(df1.corr(), annot=True)
