import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import matplotlib.ticker as ticker

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import math

import pandas_profiling

%matplotlib inline



data = pd.read_csv('/kaggle/input/shelter-animal-outcomes/train.csv.gz')
data.head()
# Check the NaN values

data.isna().sum()
data.shape
data.describe()
data.info()
# We should look for the all unique values for each columns

data['OutcomeSubtype'].unique()
data.profile_report()
# The most often outcomes is Adoption and Transfer

plt.figure(figsize=(15,5))

sns.countplot(x='OutcomeType', 

              order=data['OutcomeType'].value_counts().index, 

              data=data)

plt.show()
# Two animal types 

plt.figure(figsize=(10,5))

sns.countplot(x='AnimalType', 

              order=data['AnimalType'].value_counts().index,

              data=data)

plt.show()
# In Adoption outcome dogs prevail, in Transfer outcome cats prevail, Return to owner has almost only dogs

plt.figure(figsize=(15,8))

sns.countplot(x='OutcomeType', 

              order=data['OutcomeType'].value_counts().index, 

              hue='AnimalType',

              data=data)

plt.legend(bbox_to_anchor=(1,1))

plt.show()
# Dogs were often Adoption and Returned to owner, but Cats were ofter Transfered and Adoption

plt.figure(figsize=(15,8))

sns.countplot(x='AnimalType', 

              order=data['AnimalType'].value_counts().index, 

              hue='OutcomeType',

              data=data)

plt.legend(bbox_to_anchor=(1,1))

plt.show()
# The most frequently sex are Spayed Female and Neutered Male

plt.figure(figsize=(15,5))

sns.countplot(x='SexuponOutcome', 

              order=data['SexuponOutcome'].value_counts().index, 

              data=data)

plt.show()
# Outcome sub type has a lot of NaN values and almost only one value Partner. I think we may delete this feature

plt.figure(figsize=(25,5))

sns.countplot(x='OutcomeSubtype', 

              order=data['OutcomeSubtype'].value_counts().index,

              data=data)

plt.show()

print(data['OutcomeSubtype'].isna().sum())
plt.figure(figsize=(15,8))

sns.countplot(x='AnimalType', 

              order=data['AnimalType'].value_counts().index, 

              hue='OutcomeType',

              data=data)

plt.legend(bbox_to_anchor=(1,1))

plt.show()
data_mod = data.drop(['AnimalID', 'Name', 'OutcomeSubtype'], axis=1)
data_mod.head()
from __future__ import division

def label_age (row):

  if row['AgeuponOutcome'] == "0 years" :

      return 0

  if row['AgeuponOutcome'] == "1 year" :

      return 1

  if row['AgeuponOutcome'] == "2 years" : 

      return 2

  if row['AgeuponOutcome'] == "3 years" : 

      return 3

  if row['AgeuponOutcome'] == "4 years" : 

      return 4

  if row['AgeuponOutcome'] == "5 years" : 

      return 5

  if row['AgeuponOutcome'] == "6 years" : 

      return 6

  if row['AgeuponOutcome'] == "7 years" : 

      return 7

  if row['AgeuponOutcome'] == "8 years" : 

      return 8

  if row['AgeuponOutcome'] == "9 years" : 

      return 9

  if row['AgeuponOutcome'] == "10 years" : 

      return 10

  if row['AgeuponOutcome'] == "11 years" : 

      return 11

  if row['AgeuponOutcome'] == "12 years" : 

      return 12

  if row['AgeuponOutcome'] == "13 years" : 

      return 13

  if row['AgeuponOutcome'] == "14 years" : 

      return 14

  if row['AgeuponOutcome'] == "15 years" : 

      return 15

  if row['AgeuponOutcome'] == "16 years" :

      return 16

  if row['AgeuponOutcome'] == "17 years" :

      return 17

  if row['AgeuponOutcome'] == "18 years" :

      return 18

  if row['AgeuponOutcome'] == "20 years" :

      return 20

  if row['AgeuponOutcome'] == "1 month" :

      return 1/12

  if row['AgeuponOutcome'] == "2 months" :

      return 2/12

  if row['AgeuponOutcome'] == "3 months" :

      return 3/12

  if row['AgeuponOutcome'] == "4 months" :

      return 4/12

  if row['AgeuponOutcome'] == "5 months" :

      return 5/12

  if row['AgeuponOutcome'] == "6 months" :

      return 6/12

  if row['AgeuponOutcome'] == "7 months" :

      return 7/12

  if row['AgeuponOutcome'] == "8 months" :

      return 8/12

  if row['AgeuponOutcome'] == "9 months" :

      return 9/12

  if row['AgeuponOutcome'] == "10 months" :

      return 10/12

  if row['AgeuponOutcome'] == "11 months" :

      return 11/12

  if row['AgeuponOutcome'] == "1 week" :

      return 1/48

  if row['AgeuponOutcome'] == "1 weeks" :

      return 1/48

  if row['AgeuponOutcome'] == "2 weeks" :

      return 2/48

  if row['AgeuponOutcome'] == "3 weeks" :

      return 3/48

  if row['AgeuponOutcome'] == "4 weeks" :

      return 4/48

  if row['AgeuponOutcome'] == "5 weeks" :

      return 5/48

  if row['AgeuponOutcome'] == "1 day" :

      return 1/336

  if row['AgeuponOutcome'] == "2 days" :

      return 2/336

  if row['AgeuponOutcome'] == "3 days" :

      return 3/336

  if row['AgeuponOutcome'] == "4 days" :

      return 4/336

  if row['AgeuponOutcome'] == "5 days" :

      return 5/336

  if row['AgeuponOutcome'] == "6 days" :

      return 6/336
data_mod["Age"] = data_mod.apply(lambda row: label_age (row), axis=1)

data_mod.head()
def get_excel_date(col):

    res = pd.to_datetime(col, errors='coerce')

    return res
data_mod['DateTime'] = data_mod['DateTime'].apply(get_excel_date)

data_mod.head()
data_mod.tail()
# Divide column Datetime for two columns with month and year separate

a = pd.to_datetime(data_mod['DateTime'])

data_mod['month'] = a.dt.month

data_mod['year'] = a.dt.year

data_mod.head()
# Check for NaN values

data_mod.isna().sum()
data_mod = data_mod.drop(['DateTime', 'AgeuponOutcome'], axis=1)
data_mod = data_mod.fillna('missing')
# Let`s use column Mode value for NaN 

data_mod['OutcomeType'].replace(['missing'], 'Adoption', inplace=True) 

data_mod['Age'].replace(['missing'], '1', inplace=True) 

data_mod['SexuponOutcome'].replace(['missing'], 'Neutered Male', inplace=True) 

data_mod.isna().sum()
# Transform categoricals data

animal_type = preprocessing.LabelEncoder()

data_mod.AnimalType = animal_type.fit_transform(data_mod.AnimalType)

sex = preprocessing.LabelEncoder()

data_mod.SexuponOutcome = sex.fit_transform(data_mod.SexuponOutcome)

breed = preprocessing.LabelEncoder()

data_mod.Breed = breed.fit_transform(data_mod.Breed)

color = preprocessing.LabelEncoder()

data_mod.Color = color.fit_transform(data_mod.Color)

outcome = preprocessing.LabelEncoder()

data_mod.OutcomeType = outcome.fit_transform(data_mod.OutcomeType)

data_mod.head()
# Divide dataframe for training and testing data

ytrain = data_mod['OutcomeType']

xtrain = data_mod.drop('OutcomeType', axis=1)
ytrain.head()
xtrain.head()
# Let's see the train accuracy

rf = RandomForestClassifier(n_estimators=1000)

rf.fit(xtrain, ytrain)

tra_score=rf.score(xtrain, ytrain)
print('Training acc:', round(tra_score*100, 2), '%')