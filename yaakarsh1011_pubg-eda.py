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
train_data_loc = "../input/pubg-finish-placement-prediction/train_V2.csv"
test_data_loc = "../input/pubg-finish-placement-prediction/test_V2.csv"

train_data = pd.read_csv(train_data_loc)
test_data = pd.read_csv(test_data_loc)
train_data.info()        #Looking at the data structure of the features
train_data.head()
train_data.columns
train_data.describe()
train_data.isnull().sum() #
#In our label we have only one missing value so we can drop it
train_data = train_data.dropna()
train_data.isnull().sum()
train_data.describe()
len(train_data['matchType'].unique())
cat_data = train_data['matchType']
cat_data.shape
pd.get_dummies(cat_data).shape
train_data['matchType'].value_counts()
#let's perform frequecy encoding
matchtype_data = train_data['matchType']           #saving the matchtype data for future use
matchtype_enc = train_data['matchType'].value_counts().to_dict()        #converting the labels and their value counts into a dictionary
#it will be easier to map
train_data.matchType = train_data.matchType.map(matchtype_enc)
#replacing the matchtype column with the encoded one
train_data['matchType'].head()
train_data.info()
#let's drop these columns 
train_data = train_data.drop(columns = ['Id','groupId','matchId'])
#lets check if there is any correaltion 
#train_data.corr()['winPlacePerc'][:]
#if the data is nonrmally distributed
#train_data.corr(method = 'kendall')['winPlacePerc']
#if the data is not normally distributed
for i in train_data.columns:
    print('{} : {}'.format(i,train_data[i].skew()))
Y = train_data.winPlacePerc
X = train_data.drop(columns='winPlacePerc')
from sklearn.preprocessing import normalize
X = normalize(X)
df_X=pd.DataFrame(data=X[0:,0:],
         index=[i for i in range(X.shape[0])],
            columns=['f'+str(i) for i in range(X.shape[1])])
type(df_X)
df_Y = pd.DataFrame(Y)
type(df_Y)
df_X.head()
df_Y.head()
data_norm = pd.concat([df_X,df_Y], axis = 1, sort = False)
data_norm = data_norm.rename(columns={0:"label"})
data_norm.head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_X,df_Y,test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
Y_test
Y_pred
