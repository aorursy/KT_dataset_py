# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plot
import seaborn as sns # Beautiful plots

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train/train.csv')
df_train.columns = ['type', 'name', 'age', 'breed1', 'breed2', 'gender', 'color1', 'color2',
       'color3', 'maturity_size', 'fur_length', 'vaccinated', 'dewormed',
       'sterilized', 'health', 'quantity', 'fee', 'state', 'rescuer_id',
       'video_amt', 'description', 'pet_id', 'photo_amt', 'adoption_speed']
df_train.head()
## Sometimes is useful to know the type of the data. Most of them are integers.
df_train.dtypes
df_train.isnull().sum()
df_train.name.loc[~df_train.name.isnull()] = 1  
df_train.name.loc[ df_train.name.isnull()] = 0  
# df_train.isnull().sum()
sns.countplot(x='adoption_speed', data=df_train)
## Giving a name apparently does not impact.
sns.catplot(x='adoption_speed', col='name', kind='count', data=df_train)

## I can do the same with several other columns, just change "col" parameter. 
sns.catplot(x='adoption_speed', col='gender', kind='count', data=df_train)

## I will create the pairplot with some selected columns that I believe are promising based on visual inspection.

df_filtered = df_train.filter(['type','age','breed1', 'fur_length', 'vaccinated','sterilized','fee', 'adoption_speed' ], axis=1)
sns.pairplot(df_filtered, hue='adoption_speed')
from sklearn import tree
from sklearn.model_selection import train_test_split

## First shuffling and splitting features and target
features = df_train.sample(frac=1)
target   = features.adoption_speed
features = features[ [ 'type', 'age', 'breed1', 'fur_length', 'vaccinated', 'sterilized', 'fee']  ]

## Splitting in 
train_size      = int(len(features)*0.7)
features_train  = features[: train_size]
target_train    = target  [: train_size]
features_valid  = features[train_size :]
target_valid     = target  [train_size :]

## Classifier and fit
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(features_train, target_train)

## Predictions and evaluation
prediction_valid = clf.predict(features_valid)
accuracy = (target_valid == prediction_valid).mean()
print(accuracy)


df_test = pd.read_csv('../input/test/test.csv')
df_test.columns = ['type', 'name', 'age', 'breed1', 'breed2', 'gender', 'color1', 'color2',
       'color3', 'maturity_size', 'fur_length', 'vaccinated', 'dewormed',
       'sterilized', 'health', 'quantity', 'fee', 'state', 'rescuer_id',
       'video_amt', 'description', 'pet_id', 'photo_amt']
features_test   = df_test[ [ 'type', 'age', 'breed1', 'fur_length', 'vaccinated', 'sterilized', 'fee']  ]
prediction_test = clf.predict(features_test)

submission = pd.DataFrame(
    { 
        'PetID'         : df_test.pet_id, 
        'AdoptionSpeed' : prediction_test
    }
)
submission.set_index('PetID')
submission.to_csv('submission.csv',index=False)