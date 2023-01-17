import numpy as np

import pandas as pd



df = pd.read_csv('../input/new.csv', encoding ='iso-8859-1')
df.info()
df = df.drop(['url', 'id', 'price', 'Cid', 'DOM'], axis = 1)
df.head(5)
def str2int(s):

    try:

        return int(s)

    except:

        return np.nan

df[['livingRoom', 'drawingRoom', 'bathRoom']] = df[['livingRoom', 'drawingRoom', 'bathRoom']].applymap(str2int)
df.isnull().sum().sort_values(ascending = False).head(10)
df['buildingType'].value_counts()
df['buildingType'] = df['buildingType'].map(lambda x: x if x >= 1 else np.nan)
df['buildingType'].value_counts()
df['floor'].head()
def floorType(s):

    return s.split(' ')[0]

def floorHeight(s):

    try:

        return int(s.split(' ')[1])

    except:

        return np.nan

df['floorType'] = df['floor'].map(floorType)

df['floorHeight'] = df['floor'].map(floorHeight)
df.isnull().sum().sort_values(ascending = False).head(10)
df['floorType'].value_counts()
df['constructionTime'].value_counts()
def changeconstructionTime(s):

    if len(s) < 4:

        return np.nan

    try:

        return int(s)

    except:

        return np.nan

df['constructionTime'] = df['constructionTime'].map(changeconstructionTime)
def usedTime(buy, build):

    buy = int(buy.split('-')[0])

    try:

        return buy - build

    except:

        np.nan

df['UsedTime'] = df.apply(lambda x: usedTime(x['tradeTime'], x['constructionTime']), axis = 1)
df = df.drop('constructionTime', axis = 1)
mean_communityAverage = df['communityAverage'].mean()

df['communityAverage'] = df['communityAverage'].fillna(mean_communityAverage)
mode_col = ['buildingType', 'elevator', 'livingRoom', 'drawingRoom', 'floorHeight', 

             'fiveYearsProperty', 'subway','bathRoom', 'UsedTime']

df_mode = df[mode_col].median()

df_mode
df[mode_col] = df[mode_col].fillna(df_mode)
df.isnull().sum().sum()
y_train = df.pop('totalPrice')
str_col = ['buildingType','buildingStructure', 'renovationCondition', 'district']

df[str_col] = df[str_col].astype(str)
df['tradeTime'] = df['tradeTime'].map(lambda x: x.split('-')[0])
df = df.drop('floor',axis = 1)

df.head()
df_dummy = pd.get_dummies(df)
df_dummy.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
X = df_dummy.values

y = y_train.values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 5)
RFR = RandomForestRegressor(n_estimators=200, max_features=0.3)

RFR.fit(X_train, y_train)
y_predict = RFR.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_predict))
import matplotlib.pyplot as plt

plt.plot(y_test[:100], color = 'blue')

plt.plot(y_predict[:100], color = 'red')
from sklearn.externals import joblib

joblib.dump(RFR,'BeijingHousingPricePredicter.pkl')