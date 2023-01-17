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
import pandas as pd
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
train.info()
test.info()
train.sample(3)
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])
train.head(5)
train.tail(5)
train['Date'] = train['Date'].astype('int64')

test['Date'] = test['Date'].astype('int64')
train.iloc[235:300,:]
train.tail(5)
train.info()
from collections import defaultdict

countryCount = 0

countryList = []

provinceDict = defaultdict(list)

for country in train['Country/Region'].unique():

    countryList.append(country)

    countryCount = countryCount+1

    countryWithProvince = train[train['Country/Region'] == country]

    if countryWithProvince['Province/State'].isna().unique() == True:

        #print('No province in ', country)

        continue

    else:

        provinceDict[country].append(countryWithProvince['Province/State'].unique())

print("countryCount : ",len(countryList))

# print(countryList)

print("\nCountries with provinces :" ,len(provinceDict.keys()))

# for k,v in provinceDict.items():

#     print('\n',k,v)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def FunLabelEncoder(df):

    for c in df.columns:

        if df.dtypes[c] == object:

            le.fit(df[c].astype(str))

            #print(c)

            #df[c].fillna('khali', inplace=True)

            df[c] = le.transform(df[c].astype(str))

    return df

#trainX = FunLabelEncoder(trainX)
train = FunLabelEncoder(train)

train.info()

train.iloc[235:300,:]
test = FunLabelEncoder(test)

test.info()
train.iloc[:,:-2].sample(3)
X = train.iloc[:,:-2]

print(X.shape)

del X['Lat']

del X['Long']

print(X.shape)

X.sample(3)
Y = train.iloc[:,-2:]

print(Y.shape)

Y.sample(3)
from sklearn.model_selection import train_test_split 

trainX , valX, trainY, valY = train_test_split(X, Y, random_state=1)
y1Train = trainY.iloc[:,0]

print(y1Train.shape)

y1Train.sample(3)
y2Train = trainY.iloc[:,1]

y2Train.sample(3)
y1Val = valY.iloc[:,0]

y1Val.sample(3)
y2Val = valY.iloc[:,1]

y2Val.sample(3)
print(trainX.shape)

trainX.sample(3)
print(trainX.info())
trainX.iloc[:,1:].sample(3)
from sklearn.tree import DecisionTreeRegressor

lrModel1 = DecisionTreeRegressor(random_state = 27)

%time lrModel1.fit(trainX.iloc[:,1:], y1Train)
%time y1Pred = lrModel1.predict(valX.iloc[:,1:])

print(y1Pred[:,])
from sklearn.metrics import mean_absolute_error



print("Accuracy in train set : ", lrModel1.score(trainX.iloc[:,1:], y1Train))

print("RMSE : ", mean_absolute_error(y1Val, y1Pred)**(0.5))
lrModel2 = DecisionTreeRegressor(random_state = 27)

%time lrModel2.fit(trainX.iloc[:,1:], y2Train)



%time y2Pred = lrModel2.predict(valX.iloc[:,1:])



print("Accuracy in train set : ", lrModel2.score(trainX.iloc[:,1:], y2Train))

print("RMSE : ", mean_absolute_error(y2Val, y2Pred)**(0.5))
print(test.shape)

test.sample(3)
del test['Lat']

del test['Long']

test.sample(3)
test.iloc[:,1:].sample(3)
%time finalPred1 = lrModel1.predict(test.iloc[:,1:])

print(finalPred1[:,])
%time finalPred2 = lrModel2.predict(test.iloc[:,1:])

print(finalPred2[:,])
outputFile = pd.DataFrame({"ForecastId": test.ForecastId,

                           "ConfirmedCases": (finalPred1+0.5).astype('int'),

                           "Fatalities": (finalPred2+0.5).astype('int')})
outputFile.sample(3)
outputFile.to_csv("submission.csv", index=False)