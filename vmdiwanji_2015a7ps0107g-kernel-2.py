import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.metrics import mean_squared_error

%matplotlib inline

data = pd.read_csv('../input/train.csv')
data.head()
data.describe()
data.columns
from sklearn.preprocessing import MinMaxScaler

m = MinMaxScaler()

X = data

# X=data.loc[data['type'] == 0]

# X = x

y = X['AveragePrice']

X = X.drop(['AveragePrice'],axis=1)

X=m.fit_transform(X)

from sklearn.model_selection import train_test_split

tr_X, te_x, tr_Y, te_Y = train_test_split(X, y, test_size=0.25, random_state=42)
print(X.shape)

print()

testdata = pd.read_csv('../input/test.csv')

testdata.head()

tests = testdata

# tests=testdata.loc[testdata['type'] == 0]

tests1=testdata.loc[testdata['type'] == 1]

print(tests1.shape)

# tests= testdata.drop(['year'],axis=1)
model = xgb.XGBRegressor(n_estimators=5000,learning_rate=0.1,max_depth=7,subsample=0.8)

unoptimized_predictions = (model.fit(X, y)).predict(te_x)

acc_unop =mean_squared_error(te_Y, unoptimized_predictions)

print(acc_unop)
print(tests.shape)

# print(tests.columns)

tests=m.transform(tests)

testing_predictions = model.predict(tests)

# acc_op = mean_squared_error(te_Y, testing_predictions)

print(testing_predictions[0:5])
col1=testdata['id'].tolist()

col1=np.array(col1).astype('int32')

print(col1)

ans=np.column_stack((col1,testing_predictions))

print(ans.shape)

# print(ans1.shape)
df = pd.DataFrame(ans, columns=["id","AveragePrice"])

a=df['id'].astype('int32')

# print(a)

b=df['AveragePrice']

df=pd.concat([a,b],axis=1)

df.to_csv('list.csv', index=False)
listdata = pd.read_csv('list.csv')

listdata.head()

# listdata.shape