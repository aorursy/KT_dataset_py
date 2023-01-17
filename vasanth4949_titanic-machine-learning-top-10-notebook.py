# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("../input/machine-learning-on-titanic-data-set/train.csv")
df_train.head()
df_train.isnull().sum()
df_train["Age_Imp"] = np.where(df_train["Age"].isnull(),1,0)
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())
df_train["Cabin_Imp"] = np.where(df_train["Cabin"].isnull(),1,0) 
df_train['Cabin'].fillna('Missing',inplace=True)
df_train["Embarked"] = df_train["Embarked"].fillna(df_train["Embarked"].mode()[0])
df_train.isnull().sum()
df_train["Cabin"] = df_train["Cabin"].astype(str).str[0]
df_train.Cabin.unique()
df_train.groupby(['Cabin'])['Survived'].mean().sort_values().index
ordinal_labels=df_train.groupby(['Cabin'])['Survived'].mean().sort_values().index
ordinal_labels
ordinal_labels1 = {'T': 0, 'M': 1, 'A': 2, 'G': 3, 'C': 4, 'F': 5, 'B': 6, 'E': 7, 'D': 8}
df_train["Cabin"] = df_train["Cabin"].map(ordinal_labels1)
df_train.head()
df_train["Sex"] = pd.get_dummies(df_train["Sex"], drop_first=True)
sns.barplot(x = "Embarked", y ="Survived", data = df_train)
sns.barplot(x = "Sex", y = "Survived", data = df_train)
sns.countplot(x = "Pclass", hue = "Survived", data = df_train, color="r")
sns.barplot(x = "Pclass", y = "Fare", data = df_train)
sns.barplot(x = "Survived", y = "Fare", data = df_train)
df_train["Embarked"] = pd.get_dummies(df_train["Embarked"], drop_first=True)
sns.heatmap(df_train.corr())
df_train.corr()
#### Lets compute the Interquantile range to calculate the boundaries
IQR=df_train.Fare.quantile(0.75)-df_train.Fare.quantile(0.25)
lower_bridge=df_train['Fare'].quantile(0.25)-(IQR*3)
upper_bridge=df_train['Fare'].quantile(0.75)+(IQR*3)
print(lower_bridge), print(upper_bridge)
IQR=df_train.Age.quantile(0.75)-df_train.Age.quantile(0.25)
lower_bridge=df_train['Age'].quantile(0.25)-(IQR*3)
upper_bridge=df_train['Age'].quantile(0.75)+(IQR*3)
print(lower_bridge), print(upper_bridge)
df_train['Age']=df_train['Age'].astype(int)
df_train['Age']=df_train['Age'].astype(int)
df_train.loc[ df_train['Age'] <= 16, 'Age']= 0
df_train.loc[(df_train['Age'] > 16) & (df_train['Age'] <= 32), 'Age'] = 1
df_train.loc[(df_train['Age'] > 32) & (df_train['Age'] <= 48), 'Age'] = 2
df_train.loc[(df_train['Age'] > 48) & (df_train['Age'] <= 64), 'Age'] = 3
df_train.loc[df_train['Age'] > 64, 'Age'] = 4
df_train.drop([ "Name", "Ticket", "PassengerId",], axis=1,inplace = True)
df_train['Fare']=df_train['Fare'].astype(int)
df_train.head()
df_train.groupby(['Pclass'])['Fare'].mean().sort_values()
df_train.loc[ df_train['Fare'] <= 13, 'Fare']= 1
df_train.loc[(df_train['Fare'] > 13) & (df_train['Fare'] <= 21), 'Fare'] = 2
df_train.loc[df_train['Fare'] > 21, 'Fare'] = 3
X=df_train.drop('Survived',axis=1)
y=df_train['Survived'].astype(int)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier(max_depth=560, max_features='sqrt', min_samples_leaf=5,
                       min_samples_split=16, n_estimators=955)
model.fit(X_train , y_train)
from sklearn.metrics import accuracy_score
print(accuracy_score(model.predict(X_test) , y_test))
df1 = pd.read_csv("../input/testmod/testmod.csv")
sub= pd.read_csv("../input/machine-learning-on-titanic-data-set/gender_submission.csv")
sub
sub["Survived"] = model.predict(df1)
sub
sub.to_csv("submission1.csv", index= False)
