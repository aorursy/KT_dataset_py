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
#### Import Dependencies

%matplotlib inline

#### Start Python Imports

import math, time, random, datetime

#### Data Manipulation

import numpy as np

import pandas as pd

#### Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns

plt.style.use('seaborn-whitegrid')



#### Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize



#### Machine learning

import catboost

from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from catboost import CatBoostClassifier, Pool, cv



##### Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')

import pandas as pd 

train = pd.read_csv("../input/allstate-claims-severity/train.csv")

test = pd.read_csv("../input/allstate-claims-severity/test.csv")



#Print all rows and columns. Dont hide any

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
#Understand the data

print(train.shape)

print(test.shape)
# Save the test id for later submission 

# and drop the id from train and test. As id is unique for all rows and don't carry any information 

test_id = test["id"]

test.drop("id", axis = 1, inplace = True)

train.drop("id", axis = 1, inplace = True)

print(train.loss.describe())

print("")

print("The loss/target skewness : ",train.loss.skew())
# since the skewness of loss is about 3, which is higher. Hence it should be corrected. 

#I have choosen to go with log1p



#Before skew correction 

sns.violinplot(data=train,y="loss")  

plt.show()



train["loss"] = np.log1p(train["loss"])



#visualize the transformed column

sns.violinplot(data=train,y="loss")  

plt.show()
train.loss.describe()
data = train.copy()

valid = test.copy()

del train

del test
s = (data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(len(object_cols))

print(object_cols)
#combined the train and test so that get dummies can be applied on all at the same time. 

## But at the same time elimination due to correlation should be considered only of from train data. 

mergedata = data.append(valid)
print(mergedata.shape)

print(valid.shape)

print(len(data))

print(len(valid))
#missingno.matrix(valid, figsize = (30,12))

# no missings..
#remove the one of the pair columns which have higher abs correlation 

data_corr = data.corr()

listOfFeatures = [i for i in data_corr]

setOfDroppedFeatures = set() 

for i in range(len(listOfFeatures)) :

    for j in range(i+1,len(listOfFeatures)): #Avoid repetitions 

        feature1=listOfFeatures[i]

        feature2=listOfFeatures[j]

        if abs(data_corr[feature1][feature2]) > 0.8: #If the correlation between the features is > 0.8

            setOfDroppedFeatures.add(feature1) 
mergedata = mergedata.drop(setOfDroppedFeatures, axis=1)
s = (data.dtypes != 'object')

non_object_cols = list(s[s].index)

non_object_cols = [ col for col in non_object_cols if col not in setOfDroppedFeatures ]

print(len(non_object_cols))
nonCorrelatedWithOutput = [column for column in non_object_cols if abs(data[column].corr(data["loss"])) < 0.025]

print(len(nonCorrelatedWithOutput))

nonCorrelatedWithOutput
mergedata = mergedata.drop(nonCorrelatedWithOutput, axis=1)
mergedata.shape
# #uncomment if want to see the visualizatio of distribution (takes time)



# #names of all the columns

# cols = data.columns



# #Plot count plot for all attributes in a 29x4 grid

# n_cols = 4

# n_rows = 29

# for i in range(n_rows):

#     fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(12, 8))

#     for j in range(n_cols):

#         sns.countplot(x=cols[i*n_cols+j], data=data, ax=ax[j])

# cols taken from the visualization plots.

# to-do: find out generic method to do the same. 

unbalanced_cols = ['cat15','cat18','cat20','cat21','cat22','cat33','cat35','cat48','cat56','cat58',

                   'cat59','cat60','cat62','cat63','cat64','cat67','cat68','cat69','cat70','cat77']
mergedata = mergedata.drop(unbalanced_cols, axis=1)

print(mergedata.shape)
mergedata_ohe = pd.get_dummies(mergedata)

mergedata_ohe.shape
# seperate the train and test from mergedata

trainLen = len(data)

print("train len : ", trainLen)

train_ohe = mergedata_ohe.iloc[:trainLen]

test_ohe = mergedata_ohe.iloc[trainLen:]



del mergedata_ohe

del data

del valid



print(train_ohe.shape)

print(test_ohe.shape)
print(test_ohe.loss.isnull().sum())

# We should remove the extra added loss column to the test data

test_ohe = test_ohe.drop("loss", axis = 1)
#Build and fit the model

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import Lasso, Ridge

import xgboost as xgb # XGBoost implementation

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0007, random_state=1))

ridge = make_pipeline(RobustScaler(), Ridge(alpha =20, random_state=42))
X_train = train_ohe.drop("loss", axis = 1)

Y_train = train_ohe["loss"]

print(X_train.shape)

print(Y_train.shape)

del train_ohe
## lasso 

# lasso.fit(X_train, y_train)

# test_y_log = lasso.predict(test_ohe)

# test_y = np.exp(1)**test_y_log
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, train_size=.90,random_state=2016)





RANDOM_STATE = 2016

#'seed': RANDOM_STATE,

params = {

    'min_child_weight': 1,

    'eta': 0.03,

    'colsample_bytree': 0.5,

    'max_depth': 12,

    'subsample': 0.8,

    'alpha': 1,

    'seed': RANDOM_STATE,

    'gamma': 1.5,

    'silent': 1,

    'verbose_eval': True,

    'nthread':7,

    'base_score':7.76

}







xgtrain = xgb.DMatrix(x_train, label=y_train)

xgval = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [ (xgtrain,'train'),(xgval,'eval')]



model = xgb.train(params, xgtrain,300, watchlist,verbose_eval=True)



del xgtrain

del xgval

xgtest = xgb.DMatrix(test_ohe)

test_y_log = model.predict(xgtest)

test_y = np.exp(1)**test_y_log
del X_train

del y_train
submission = pd.DataFrame()

submission["id"] = test_id

print(test_y.shape)

print(test_id.shape)



submission["loss"] = test_y

submission.to_csv("submission_svm_l.csv", index=False)
# score is : 1126.77