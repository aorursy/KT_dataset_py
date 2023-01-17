import numpy as np 

import pandas as pd

import plotly.express as px
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.info()
train.describe()
train.shape
train.corr()['SalePrice'].sort_values(ascending=False)
fig = px.box(train['SalePrice'])

fig.update_layout(title='Sale Price Distribution',title_x=0.5)
fig = px.histogram(train['SalePrice'])

fig.update_layout(title='Sale Price Distribution',title_x=0.5)

fig.show()
fig = px.scatter(train, x="GarageArea", y="SalePrice", trendline="ols")

fig.update_layout(title='Regression plot of Garage Area and Sale Price',title_x=0.5)

fig.show()
sale_types = train.groupby('SaleType')[['SalePrice']].sum().reset_index().sort_values('SalePrice',ascending=False)

fig = px.bar(sale_types,x=sale_types['SaleType'],y=sale_types['SalePrice'])

fig.update_layout(title='Sale Prices by Sale Type',title_x=0.5)

fig.show()
house_style = train.groupby('HouseStyle')[['SalePrice']].sum().reset_index().sort_values('SalePrice',ascending=False)

fig = px.bar(sale_types,x=house_style['HouseStyle'],y=house_style['SalePrice'],labels={'x':'House Style','y':'Cumulative Sales Price'})

fig.update_layout(title='Sale Prices by House Style',title_x=0.5)

fig.show()
house_style = train.groupby('SaleCondition')[['SalePrice']].sum().reset_index().sort_values('SalePrice',ascending=False)

fig = px.bar(sale_types,x=house_style['SaleCondition'],y=house_style['SalePrice'],labels={'x':'Sale Condition','y':'Cumulative Sales Price'})

fig.update_layout(title='Sale Prices by Sale Condition',title_x=0.5)

fig.show()
house_style = train.groupby('LandSlope')[['SalePrice']].sum().reset_index().sort_values('SalePrice',ascending=False)

fig = px.bar(sale_types,x=house_style['LandSlope'],y=house_style['SalePrice'],labels={'x':'Land Slope','y':'Cumulative Sales Price'})

fig.update_layout(title='Sale Prices by Land Slope',title_x=0.5)

fig.show()
def clean_dataset(df):

    

    df = df.drop(['MiscFeature','Fence','PoolQC','FireplaceQu','Alley','LowQualFinSF'],axis=1).fillna(method='ffill')

    df = pd.get_dummies(df,columns=df.loc[:,df.dtypes==object].columns,drop_first=True)

    return df

train = clean_dataset(train)

test = clean_dataset(test)
selected_values = list(test.columns)

selected_values.append('SalePrice')



train = train[selected_values]
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import KFold
X = train.drop(['SalePrice','Id'],axis=1)

y = train['SalePrice']
kf = KFold(random_state=101,shuffle=True,n_splits=10,)
for train_index,test_index in kf.split(X,y):

    X_train,X_test = X.iloc[train_index],X.iloc[test_index],

    y_train,y_test = y.iloc[train_index],y.iloc[test_index]
regressor=XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,

       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,

       n_jobs=1, nthread=None, objective='reg:squarederror', random_state=101,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, subsample=1)
regressor.fit(X_train,y_train)
regressor.score(X_train,y_train)
regressor.score(X_test,y_test)
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test,regressor.predict(X_test)))
predictions = regressor.predict(test.drop('Id',axis=1))
submission = pd.DataFrame(data=predictions,index=test.index,columns=['SalePrice']).reset_index()

submission.columns = ['Id','SalePrice']

submission['Id'] = test['Id']

submission.to_csv('submission.csv',index=False)