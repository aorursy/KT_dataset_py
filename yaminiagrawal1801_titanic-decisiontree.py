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
gender_submission_df = pd.read_csv("//kaggle//input//titanic//gender_submission.csv")
test_df = pd.read_csv("//kaggle//input//titanic//test.csv")
train_df = pd.read_csv("//kaggle//input//titanic//train.csv")
train_df.head()
test_df.head()
train_df.columns
train_df.isna().sum()
train_df['Age']=train_df['Age'].fillna(0)
train_df['Survived'].unique()
train_df['Sex'].unique()
train_df['Embarked'].unique()
train_df['Embarked']=train_df['Embarked'].fillna('X')
train_df['Embarked'].unique()
test_df.columns
test_df.isna().sum()
test_df['Age']=test_df['Age'].fillna(0)
test_df['Age'].isnull().sum()
test_df['Fare']=test_df['Fare'].fillna(0)
test_df['Fare'].isnull().sum()
test_df['Embarked'].unique()
train_X = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']].values
train_X[0:5]
test_X = test_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']].values
test_X[0:5]
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['male','female'])
train_X[:,1] = le_sex.transform(train_X[:,1]) 
test_X[:,1] = le_sex.transform(test_X[:,1]) 

le_Embarked = preprocessing.LabelEncoder()
le_Embarked.fit(['S', 'C', 'Q','X'])
train_X[:,4] = le_Embarked.transform(train_X[:,4])
test_X[:,4] = le_Embarked.transform(test_X[:,4])


train_X[0:5]
test_X[0:5]
train_y = train_df["Survived"]
train_y[0:5]
from sklearn.tree import DecisionTreeClassifier
survivalTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
survivalTree # it shows the default parameters
survivalTree.fit(train_X,train_y)
predTree = survivalTree.predict(test_X)
print(gender_submission_df[0:5])
print(predTree[0:5])
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predTree})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(gender_submission_df['Survived'], predTree))
