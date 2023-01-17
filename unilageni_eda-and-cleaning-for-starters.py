# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline

import numpy as np

import pandas as pd

import gc

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from scipy.stats import rankdata



print(os.listdir("../input"))
train = pd.read_csv("../input/intercampusai2019/train.csv")

test = pd.read_csv("../input/intercampusai2019/test.csv")

print("Number of rows and columns in train set :",train.shape)

print("Number of rows and columns in test set :",test.shape)
train.head()
test.head()
sns.countplot(train['Promoted_or_Not'], palette='Set3')

train.Promoted_or_Not.value_counts()
train.describe()
test.describe()

train.isnull().sum()
feats = [f for f in train.columns if f not in ['EmployeeNo','Promoted_or_Not']]

for i in feats:

    print ('==' + str(i) + '==')

    print ('train:' + str(train[i].nunique()/train.shape[0]))

    print ('test:' + str(test[i].nunique()/test.shape[0]))
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train[features].std(axis=1),color="green", kde=True,bins=120, label='train')

sns.distplot(test[features].std(axis=1),color="blue", kde=True,bins=120, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(train[features].mean(axis=0),color="yellow",kde=True,bins=50, label='train')

sns.distplot(test[features].mean(axis=0),color="red", kde=True,bins=50, label='test')

plt.legend()

plt.show()
plt.figure(figsize=(16,6))

features = train[feats].columns.values

plt.title("Distribution of std values per row in the train and test set")

sns.distplot(train[features].std(axis=0),color="red", kde=True,bins=50, label='train')

sns.distplot(test[features].std(axis=0),color="yellow", kde=True,bins=50, label='test')

plt.legend()

plt.show()
correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

correlations.head(10)
correlations.tail(10)
feats_target = [f for f in train.columns if f not in ['EmployeeNo']]

correlations = train[feats_target].corr().abs().unstack().sort_values(kind="quicksort").reset_index()

correlations = correlations[correlations['level_0'] != correlations['level_1']]

corr = correlations[correlations['level_0']=='Promoted_or_Not']

corr.head()
# creating a copy of train

data = train.copy()
data.Promoted_or_Not.value_counts()
data.isnull().sum()

# we have null values in the qualification tables lets clean that up
# so what i did here was fill in the null values in the qualification features with no qualification

data.Qualification = data.Qualification.fillna("no qualification")

test.Qualification  = test.Qualification.fillna("no qualification")
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
data.head()
TARGET = np.array( list(data['Promoted_or_Not'].values) )

data.shape
TARGET.shape
features = [x for x in data.columns if x not in ["Promoted_or_Not","EmployeeNo"]]
# model.predict_proba(test)
##what is happening here is that i am combinning both the train and test together

ntrain = data.shape[0]

ntest = test.shape[0]



all_data = pd.concat((data, test)).reset_index(drop=True)

print("all_data size is : {}".format(all_data.shape))
#feature engineering 

all_data["age"] = all_data.Year_of_recruitment - all_data.Year_of_birth

all_data.drop("Year_of_birth", axis  = 1,inplace  = True)
all_data.Channel_of_Recruitment.unique()
# label encoding some of the categorical features

all_data.Channel_of_Recruitment = pd.DataFrame(label.fit_transform(all_data.Channel_of_Recruitment))

all_data.Division  =pd.DataFrame(label.fit_transform(all_data.Division))

all_data.State_Of_Origin = pd.DataFrame(label.fit_transform(all_data.State_Of_Origin))
all_data.dtypes
all_data.head()
all_data.drop('EmployeeNo',axis = 1,inplace =True)
#import geocoder

all_data = pd.get_dummies(all_data)



# remove constant features

[feat for feat in all_data.columns if all_data[feat].std() == 0]

## no constant features :)



#Get the new dataset

data = all_data[:ntrain]

test = all_data[ntrain:]
test.head()
test.drop("Promoted_or_Not",axis = 1,inplace = True)
sample = pd.read_csv("../input/intercampusai2019/sample_submission2.csv")

sample.head()
data.shape
# splitting the training features into test and train for valisdation

X= data.drop( 'Promoted_or_Not', axis = 1)

y = data["Promoted_or_Not"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# data =  data.drop("Promoted_or_Not",axis =1)


from catboost import CatBoostClassifier

model=CatBoostClassifier(iterations=1000, depth=7, learning_rate=0.05,eval_metric="Accuracy")

model.fit(X_train,y_train)

predictions  = model.predict(test).astype(int)
test = pd.read_csv("../input/intercampusai2019/test.csv")

sample.head()
sample.EmployeeNo = test.EmployeeNo
sample.Promoted_or_Not = predictions
sample.head()
sample.to_csv("low1.csv",index = False)