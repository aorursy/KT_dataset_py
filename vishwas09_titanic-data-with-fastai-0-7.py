!pip install fastai==0.7.0 --no-deps

!pip install torch==0.4.1 torchvision==0.2.1
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
from fastai import *

from fastai.tabular import *

from fastai.vision import *
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.isnull().sum()
df_test.info()

df_train.Cabin.fillna("N", inplace=True)

df_test.Cabin.fillna("N", inplace=True)
print (df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
Y = pd.Series(df_train['Survived'])

df_train["Fare"].fillna(df_train["Fare"].median(), inplace=True)

df_test["Fare"].fillna(df_test["Fare"].median(), inplace=True)
train_title=[i.split(",")[1].split(".")[0].strip() for i in df_train["Name"]]

df_train["Title"]=pd.Series(train_title)

df_train.head()
test_title=[i.split(",")[1].split(".")[0].strip() for i in df_test["Name"]]

df_test["Title"]=pd.Series(test_title)

df_test.describe()