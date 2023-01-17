import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection, preprocessing

import xgboost as xgb

import datetime



#load files

train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])

test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])

id_test = test.id
Q1 = np.percentile(np.log1p(train.price_doc.values), 25)

Q2 = np.percentile(np.log1p(train.price_doc.values), 50)

Q3 = np.percentile(np.log1p(train.price_doc.values), 75)

IQR=Q3-Q1

infbdd=Q1-1.5 * IQR 

supbdd=Q3+1.5 * IQR 

train['price_doc'].ix[train['price_doc']>int(2.5*np.exp(supbdd))] = int(2.5*np.exp(supbdd))

train['price_doc'].ix[train['price_doc']<int(np.exp(13.5))] = int(np.exp(13.5))
equal_index = [601,1896,2791]

test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]
bad_index = train[train.life_sq < 5].index

train.ix[bad_index, "life_sq"] = np.NaN

bad_index = test[test.life_sq < 5].index

test.ix[bad_index, "life_sq"] = np.NaN

bad_index = train[train.full_sq < 5].index

train.ix[bad_index, "full_sq"] = np.NaN

bad_index = test[test.full_sq < 5].index

test.ix[bad_index, "full_sq"] = np.NaN
bad_index = test[test.life_sq > test.full_sq]

bad_index

exchange_index=[64,119,171]

life_bad_index=[2027, 2031, 5187]

full_bad_index=[2804]

test.ix[life_bad_index, "life_sq"] = np.NaN

test.ix[full_bad_index, "full_sq"] = np.NaN

for cat in exchange_index:

    dog=test.ix[cat, "life_sq"]

    test.ix[cat, "life_sq"] = test.ix[cat, "full_sq"]

    test.ix[cat, "full_sq"]=dog

    
test.ix[exchange_index]
print(np.nanpercentile(np.log1p(train.price_doc.values)/np.log1p(train.full_sq.values), 99.9))

print(np.nanpercentile(np.log1p(train.price_doc.values)/np.log1p(train.full_sq.values), 0.1))

print(np.nanpercentile(np.log1p(train.price_doc.values)/np.log1p(train.life_sq.values), 99.9))

print(np.nanpercentile(np.log1p(train.price_doc.values)/np.log1p(train.life_sq.values), 0.1))

print(np.nanpercentile(train.life_sq.values/train.full_sq.values, 0.1))

print(np.nanpercentile(test.life_sq.values/test.full_sq.values, 0.1))
#life_bad_index=train[(train.life_sq > train.full_sq) & ((np.log1p(train.price_doc.values)/np.log1p(train.full_sq.values))>3)&((np.log1p(train.price_doc.values)/np.log1p(train.full_sq.values))<5.25)].index

#train.ix[life_bad_index, "life_sq"] = np.NaN

#full_bad_index=train[(train.life_sq > train.full_sq)&((np.log1p(train.price_doc.values)/np.log1p(train.life_sq.values))>3.12)&((np.log1p(train.price_doc.values)/np.log1p(train.life_sq.values))<6.46)].index

#train.ix[full_bad_index, "full_sq"] = np.NaN

bad_index = train[train.life_sq > train.full_sq].index

train.ix[life_bad_index, "life_sq"] = np.NaN
#life_bad_index=train[(train.life_sq / train.full_sq<0.3) & ((np.log1p(train.price_doc.values)/np.log1p(train.full_sq.values))>3)&((np.log1p(train.price_doc.values)/np.log1p(train.full_sq.values))<5.25)].index

#train.ix[life_bad_index, "life_sq"] = np.NaN

#full_bad_index=train[(train.life_sq / train.full_sq<0.3)&((np.log1p(train.price_doc.values)/np.log1p(train.life_sq.values))>3.12)&((np.log1p(train.price_doc.values)/np.log1p(train.life_sq.values))<6.46)].index

#train.ix[full_bad_index, "full_sq"] = np.NaN

bad_index = train[(train.life_sq / train.full_sq)<0.3].index

train.ix[life_bad_index, "life_sq"] = np.NaN
bad_index = test[(test.life_sq / test.full_sq)<0.27]

bad_index
bad=train[((np.log1p(train.price_doc.values)/np.log1p(train.life_sq.values))<3.13)]

bad
train=train[((np.log1p(train.price_doc.values)/np.log1p(train.life_sq.values))>3.13)]

train=train[((np.log1p(train.price_doc.values)/np.log1p(train.life_sq.values)) <6.35)]

plt.figure(figsize=(12,12))

sns.jointplot(x=np.log1p(train.life_sq.values), y=np.log1p(train.price_doc.values), size=10,kind="hex")

plt.ylabel('Log of Price', fontsize=12)

plt.xlabel('Log of living area in square metre', fontsize=12)

plt.show()
bad=train[train.full_sq==0]

bad
bad_index = train[train.life_sq > 300]

bad_index
bad_index = test[test.life_sq > 200]

bad_index

#test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad=test[test.floor<60]

bad
bad_index = test[test.floor > test.max_floor].index

bad
test.max_floor.describe(percentiles= [0.9999])


bad_index = train[train.floor == 0].index

train.ix[bad_index, "floor"] = np.NaN

bad_index = train[train.max_floor == 0].index

train.ix[bad_index, "max_floor"] = np.NaN

bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values]

bad_index
bad_index = test[test.max_floor==0]

bad_index