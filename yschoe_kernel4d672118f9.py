import os



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# 경고문 무시(seaborn)

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# price

df_train['price'] = np.log1p(df_train['price'])



from datetime import date

# year

df_train['year'] = df_train['date'].apply(lambda x : str(x[:4])).astype(int)

df_test['year'] = df_test['date'].apply(lambda x : str(x[:4])).astype(int)

# month

df_train['month'] = df_train['date'].apply(lambda x : str(x[4:6])).astype(int)

df_test['month'] = df_test['date'].apply(lambda x : str(x[4:6])).astype(int)

# day

df_train['day'] = df_train['date'].apply(lambda x : str(x[6:8])).astype(int)

df_test['day'] = df_test['date'].apply(lambda x : str(x[6:8])).astype(int)



df_train['age'] = df_train[['year']].sub(df_train['yr_built'], axis=0)

df_test['age'] = df_test[['year']].sub(df_test['yr_built'], axis=0)



del df_train['date']

del df_test['date']



for index, row in df_train.iterrows():

    delta = (date(df_train.at[index, 'year'], df_train.at[index, 'month'], df_train.at[index, 'day']) - date(1970, 1, 1))

    df_train.at[index, 'days'] = delta.days

for index, row in df_test.iterrows():

    delta = (date(df_test.at[index, 'year'], df_test.at[index, 'month'], df_test.at[index, 'day']) - date(1970, 1, 1))

    df_test.at[index, 'date'] = delta.days

    

del df_train['year']

del df_train['month']

#del df_train['day']



del df_test['year']

del df_test['month']

#del df_test['day']



# sqft_living

del df_train['sqft_living']

del df_test['sqft_living']



df_train['rooms'] = df_train['bedrooms'] + df_train['bathrooms']

df_test['rooms'] = df_test['bedrooms'] + df_test['bathrooms']



df_train['is_renovated'] = df_train['yr_renovated'].apply(lambda x: 0 if x==0 else 1)

df_test['is_renovated'] = df_test['yr_renovated'].apply(lambda x: 0 if x==0 else 1)



# yr_renovated

for index, row in df_train.iterrows():

    if row['yr_renovated'] == 0:

        df_train.at[index, 'yr_renovated'] = df_train.at[index, 'yr_built']



for index, row in df_test.iterrows():

    if row['yr_renovated'] == 0:

        df_test.at[index, 'yr_renovated'] = df_test.at[index, 'yr_built']

del df_train['yr_built']

del df_test['yr_built']



# zipcode

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(df_train['zipcode'])

le.fit(df_test['zipcode'])



df_train['zipcode'] = le.transform(df_train['zipcode'])

df_test['zipcode'] = le.transform(df_test['zipcode'])
X_train = df_train.loc[:, ~df_train.columns.isin(['id', 'price'])]

y_train = df_train['price']

X_test = df_test.loc[:, ~df_test.columns.isin(['id', 'price'])]
from sklearn.model_selection import cross_val_score, KFold

from sklearn.ensemble import GradientBoostingRegressor



kfold = KFold(n_splits=5)

gbr = GradientBoostingRegressor(learning_rate = 0.1,

                                max_depth = 5,

                                max_features = 6,

                                n_estimators = 800,

                                subsample = 1)

scores = cross_val_score(gbr, X_train.values, y_train, cv=kfold)

rmse = np.sqrt(scores)

print("교차 검증 점수: {}".format(scores))

print("교차 검증 평균 점수: {}".format(scores.mean()))

print("RMSE: {}".format(rmse.mean()))
gbr.fit(X_train, y_train)

pred = gbr.predict(X_test)

pred = np.expm1(pred)

df_submit =  pd.DataFrame(data={'id':df_test['id'],'price':pred})

df_submit.to_csv('submission.csv', index=False)

print("complete!")