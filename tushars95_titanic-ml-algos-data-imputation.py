# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/train.csv")
sample_submission = pd.read_csv("../input/gender_submission.csv")
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
# Any results you write to the current directory are saved as output.
data.head()
import pandas_profiling as pf
pf.ProfileReport(data)
data['Survived'].value_counts()
from sklearn.model_selection import train_test_split

data_dummies=pd.get_dummies(train.drop(['Ticket','Name','Cabin'], axis=1))

train, validate=train_test_split(data_dummies,test_size=0.2,random_state=100)

train_x=train.drop('Survived',axis=1)

validate_x=validate.drop('Survived',axis=1)

train_y=train['Survived']

validate_y=validate['Survived']
# method 1 : Impute with column mean
data_dummies['Age_m1'] = data_dummies['Age'].fillna(
    data_dummies['Age'].mean())
# Data_Dummies['Age'].mean()

#Method 2 : Impute with column median (if you have outlier in Age)

data_dummies['Age_m2'] = data_dummies['Age'].fillna(
    data_dummies['Age'].median())
pd.isnull(data_dummies['Age_m1']).sum()

#Method 3 :  impute using other columns
data_dummies[pd.isnull(data_dummies['Age'])]
