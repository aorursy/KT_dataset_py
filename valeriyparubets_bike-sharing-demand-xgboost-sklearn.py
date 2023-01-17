import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from datetime import datetime
from scipy import stats
trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')
trainData.head(2)
fig, axes = plt.subplots(figsize=(15, 4), ncols=2, nrows=1)
sn.distplot(trainData["count"],ax=axes[0])
plt.plot(pd.rolling_mean(trainData['count'], 100))
plt.show()
trainData['logcount'] = trainData['count'].apply(lambda x: np.log1p(x))
fig, axes = plt.subplots(figsize=(15, 8))
sn.distplot(trainData["logcount"], ax=axes)
trainData['date'] = trainData.datetime.apply(lambda x : x.split()[0])
trainData['hour'] = trainData.datetime.apply(lambda x : x.split()[1].split(":")[0])
trainData['weekday'] = trainData.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())
trainData['month'] = trainData.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)

testData['date'] = testData.datetime.apply(lambda x : x.split()[0])
testData['hour'] = testData.datetime.apply(lambda x : x.split()[1].split(":")[0])
testData['weekday'] = testData.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())
testData['month'] = testData.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)

timeColumn = testData['datetime']
import xgboost as xgb

X = trainData.drop(['count', 'datetime', 'registered', 'casual', 'date', 'logcount'], axis=1).values
Y = trainData['logcount'].values

testX = testData.drop(['datetime', 'date'], axis=1).values

trainMatrix = xgb.DMatrix(X, label=Y)

max_depth = 5
min_child_weight = 8
subsample = 0.9
num_estimators = 1000
learning_rate = 0.1

clf = xgb.XGBRegressor(max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                n_estimators=num_estimators,
                learning_rate=learning_rate)

clf.fit(X,Y)

pred = clf.predict(testX)
pred = np.expm1(pred)

submission = pd.DataFrame({
        "datetime": timeColumn,
        "count": pred
    })
submission.to_csv('XGBNoFE.csv', index=False)
fig, axes = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(15, 8)
sn.boxplot(data=trainData, y='count', x='season', ax=axes[0])
sn.boxplot(data=trainData, y='count', x='workingday', ax=axes[1])
axes[0].set(xlabel='season', ylabel='count')
axes[1].set(xlabel='workingday', ylabel='count')
fix, axes = plt.subplots(figsize=(15, 10))
sn.boxplot(data=trainData, y='count', x='hour', ax=axes)
trainDataWithoutOutliers = trainData[np.abs(trainData['count']-trainData['count'].mean())
                                     <=(3*trainData['count'].std())] 
print(trainDataWithoutOutliers.shape)
trainData = trainDataWithoutOutliers
corrMat = trainData.corr()
mask = np.array(corrMat)
mask[np.tril_indices_from(mask)] = False
fig, ax= plt.subplots(figsize=(20, 10))
sn.heatmap(corrMat, mask=mask,vmax=1., square=True,annot=True)
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 15))

meanMonthly = pd.DataFrame(trainData.groupby('month')['count'].mean()).reset_index().sort_values(by='count', ascending=False)
sn.barplot(data=meanMonthly, x='month', y='count', ax=axes[0])
axes[0].set(xlabel='month', ylabel='count')

hoursSeasonly = pd.DataFrame(trainData.groupby(['hour', 'season'], sort=True)['count'].mean()).reset_index()
sn.pointplot(x=hoursSeasonly['hour'], y=hoursSeasonly['count'], hue=hoursSeasonly['season'], data=hoursSeasonly, join=True, ax=axes[1])
axes[1].set(xlabel='hour', ylabel='count')

hoursDayly = pd.DataFrame(trainData.groupby(['hour','weekday'], sort=True)['count'].mean()).reset_index()
sn.pointplot(x=hoursDayly['hour'], y=hoursDayly['count'], hue=hoursDayly['weekday'], data=hoursDayly, join=True,ax=axes[2])
axes[2].set(xlabel='hour', ylabel='count')

hoursSeasonly = pd.DataFrame(trainData.groupby(['hour', 'month'], sort=True)['count'].mean()).reset_index()
sn.pointplot(x=hoursSeasonly['hour'], y=hoursSeasonly['count'], hue=hoursSeasonly['month'], data=hoursSeasonly, join=True, ax=axes[3])
axes[1].set(xlabel='hour', ylabel='count')
X = trainData.drop(['date', 'temp', 'casual', 'registered', 'logcount', 'datetime', 'count'], axis=1)

season_df = pd.get_dummies(trainData['season'], prefix='s', drop_first=True)
weather_df = pd.get_dummies(trainData['weather'], prefix='w', drop_first=True)
hour_df = pd.get_dummies(trainData['hour'], prefix='h', drop_first=True)
weekday_df = pd.get_dummies(trainData['weekday'], prefix='d', drop_first=True)
month_df = pd.get_dummies(trainData['month'], prefix='m', drop_first=True)

X = X.join(season_df)
X = X.join(weather_df)
X = X.join(hour_df)
X = X.join(weekday_df)
X = X.join(month_df)

X = X.values
Y=trainData['logcount'].values
print(X.shape)

testX = testData.drop(['date', 'temp', 'datetime'], axis=1)

season_df = pd.get_dummies(testData['season'], prefix='s', drop_first=True)
weather_df = pd.get_dummies(testData['weather'], prefix='w', drop_first=True)
hour_df = pd.get_dummies(testData['hour'], prefix='h', drop_first=True)
weekday_df = pd.get_dummies(testData['weekday'], prefix='d', drop_first=True)
month_df = pd.get_dummies(testData['month'], prefix='m', drop_first=True)

testX = testX.join(season_df)
testX = testX.join(weather_df)
testX = testX.join(hour_df)
testX = testX.join(weekday_df)
testX = testX.join(month_df)

testX = testX.values
print(testX.shape)
clf=xgb.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4,colsample_bytree=0.6,subsample=0.6)
clf.fit(X,Y)

pred = clf.predict(testX)
pred = np.expm1(pred)

submission = pd.DataFrame({
        "datetime": timeColumn,
        "count": pred
    })
submission.to_csv('XGBwithFE.csv', index=False)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer

def loss_func(truth, prediction):
    truth = np.expm1(truth)
    prediction = np.expm1(prediction)
    log1 = np.array([np.log(x + 1) for x in truth])
    log2 = np.array([np.log(x + 1) for x in prediction])
    return np.sqrt(np.mean((log1 - log2)**2))
param_grid = {
    'n_estimators': [50, 80, 100, 120],
    'max_depth': [None, 1, 2, 5],
    'max_features': ['sqrt', 'log2', 'auto']
}

scorer = make_scorer(loss_func, greater_is_better=False)

regr = RandomForestRegressor(random_state=42)

rfr = GridSearchCV(regr, param_grid, cv=4, scoring=scorer, n_jobs=4).fit(X, Y)
print('\tParams:', rfr.best_params_)
print('\tScore:', rfr.best_score_)
pred = rfr.predict(testX)
pred = np.expm1(pred)

submission = pd.DataFrame({
        "datetime": timeColumn,
        "count": pred
    })
submission.to_csv('RandomForest.csv', index=False)
#
#param_grid = {
#    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
#    'n_estimators': [100, 1000, 1500, 2000, 4000],
#    'max_depth': [1, 2, 3, 4, 5, 8, 10]
#}
#
#scorer = make_scorer(loss_func, greater_is_better=False)
#
#gb = GradientBoostingRegressor(random_state=42)
#
#gbr = GridSearchCV(gb, param_grid, cv=4, scoring=scorer, n_jobs=3).fit(X, Y)
#print('\tParams:', gbr.best_params_)
#print('\tScore:', gbr.best_score_)

gbr = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01, max_depth=4)

gbr.fit(X, Y)
pred = gbr.predict(testX)
pred = np.expm1(pred)

submission = pd.DataFrame({
        "datetime": timeColumn,
        "count": pred
    })
submission.to_csv('GradientBoost.csv', index=False)