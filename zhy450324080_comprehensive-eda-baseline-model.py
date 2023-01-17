import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, metrics
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 1000) 

pd.set_option('display.max_rows', 200) 

pd.set_option('display.max_columns', 200) 

df_train =  pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.info()
df_train['Id'].nunique()
df_train['groupId'].nunique()
df_train['matchId'].nunique()
#================== EDA =======================================================

# ---------- single distributions ---------

plt.hist(df_train['winPlacePerc'])
plt.xlabel("winPlacePerc") 
plt.ylabel("count") 
plt.title('Distribution of winPlacePerc')
plt.figure(figsize=[10,6])
df_train['assists'].value_counts().plot(kind='bar')
plt.title("Distribution of assists") 
plt.ylabel("count") 
plt.show()
print(df_train['assists'].value_counts())
plt.figure(figsize=[10,6])
df_train['boosts'].value_counts().plot(kind='bar')
plt.title("Distribution of boosts") 
plt.ylabel("count") 
plt.show()
print(df_train['boosts'].value_counts())
plt.figure(figsize=[10,6])
(df_train.loc[df_train['damageDealt']>500, 'damageDealt'].astype(float)).value_counts().plot(kind='bar')
plt.title("Distribution of damageDealt") 
plt.ylabel("count") 
plt.show()

plt.figure(figsize=[10,6])
df_train['DBNOs'].value_counts().plot(kind='bar')
plt.title("Distribution of DBNOs") 
plt.ylabel("count") 
plt.show()
print(df_train['DBNOs'].value_counts())
plt.figure(figsize=[10,6])
df_train['headshotKills'].value_counts().plot(kind='bar')
plt.title("Distribution of headshotKills") 
plt.ylabel("count") 
plt.show()
print(df_train['headshotKills'].value_counts())
plt.figure(figsize=[10,6])
df_train['heals'].value_counts().plot(kind='bar')
plt.title("Distribution of heals") 
plt.ylabel("count") 
plt.show()
print(df_train['heals'].value_counts())
plt.figure(figsize=[18,4])
df_train['killPlace'].value_counts().plot(kind='bar')
plt.title("Distribution of killPlace") 
plt.ylabel("count") 
plt.show()
print(df_train['killPlace'].value_counts())
df_train['matchId'].nunique()
plt.figure(figsize=[18,4])
df_train['kills'].value_counts().plot(kind='bar')
plt.title("Distribution of kills") 
plt.ylabel("count") 
plt.show()
print(df_train['kills'].value_counts())

plt.figure(figsize=[18,4])
df_train['killStreaks'].value_counts().plot(kind='bar')
plt.title("Distribution of killStreaks") 
plt.ylabel("count") 
plt.show()
print(df_train['killStreaks'].value_counts())
plt.figure(figsize=[18,4])
df_train['maxPlace'].value_counts().plot(kind='bar')
plt.title("Distribution of maxPlace") 
plt.ylabel("count") 
plt.show()
print(df_train['maxPlace'].value_counts())
plt.figure(figsize=[18,4])
df_train['numGroups'].value_counts().plot(kind='bar')
plt.title("Distribution of numGroups") 
plt.ylabel("count") 
plt.show()
print(df_train['numGroups'].value_counts())
print(df_train['revives'].value_counts())
plt.figure(figsize=[10,6])
(df_train.loc[df_train['longestKill']>0, 'longestKill'].astype(float)).value_counts().plot(kind='bar')
plt.title("Distribution of longestKill") 
plt.ylabel("count") 
plt.show()
print(df_train['teamKills'].value_counts())
# teamKills
# is
# not 
# a 
# valuable
# variable
print(df_train['vehicleDestroys'].value_counts())
print(df_train['weaponsAcquired'].value_counts())
# ---------------- correlation --------------

# variable correlation 
correlation = df_train.corr()
correlation = correlation['winPlacePerc'].sort_values(ascending=False)
print(correlation.head(20))
sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
fig=plt.gcf()
fig.set_size_inches(20,16)
plt.show()
train_ = df_train

def show_count_sum(df, col,n=10):
    return df.groupby(col).agg({'winPlacePerc': ['count', 'mean']}).sort_values(('winPlacePerc', 'count'), ascending=False).head(n)
show_count_sum(train_, 'assists')
show_count_sum(train_, 'boosts')
show_count_sum(train_, 'DBNOs')
show_count_sum(train_, 'headshotKills')
show_count_sum(train_, 'heals')
show_count_sum(train_, 'killPlace')
show_count_sum(train_, 'killPoints')
show_count_sum(train_, 'kills')
show_count_sum(train_, 'killStreaks')
show_count_sum(train_, 'maxPlace')
show_count_sum(train_, 'numGroups')
show_count_sum(train_, 'revives')
show_count_sum(train_, 'vehicleDestroys')
show_count_sum(train_, 'weaponsAcquired')
show_count_sum(train_, 'winPoints')
data = pd.concat([train_['damageDealt'], train_['winPlacePerc']], axis=1)
data.plot.scatter(x='damageDealt', y='winPlacePerc')
data = pd.concat([train_['killPoints'], train_['winPlacePerc']], axis=1)
data.plot.scatter(x='killPoints', y='winPlacePerc')

data = pd.concat([train_['longestKill'], train_['winPlacePerc']], axis=1)
data.plot.scatter(x='longestKill', y='winPlacePerc')

data = pd.concat([train_['rideDistance'], train_['winPlacePerc']], axis=1)
data.plot.scatter(x='rideDistance', y='winPlacePerc')
data = pd.concat([train_['walkDistance'], train_['winPlacePerc']], axis=1)
data.plot.scatter(x='walkDistance', y='winPlacePerc')
data = pd.concat([train_['winPoints'], train_['winPlacePerc']], axis=1)
data.plot.scatter(x='winPoints', y='winPlacePerc')
#====================== Predicting ============================================

Y = (df_train['winPlacePerc'].astype(float)).values

sum_id = df_test["Id"].values

df_train = df_train.drop(['Id','groupId','matchId','winPlacePerc'], axis = 1)
                          
df_test= df_test.drop(['Id','groupId','matchId'], axis = 1)
'''
lgb_pred[lgb_pred > 1] = 1
    
test  = pd.read_csv('../input/test.csv')
test['winPlacePercPred'] = lgb_pred
aux = test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
    
subm = test[['Id','winPlacePerc']]
    
subm.to_csv("LGB.csv", index=False)
'''
#=========================== lgb =================================== 

import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_lgb.fit(df_train, Y)
lgb_pred = model_lgb.predict(df_test)

lgb_pred[lgb_pred > 1] = 1
    
test  = pd.read_csv('../input/test.csv')
test['winPlacePercPred'] = lgb_pred
aux = test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
    
subm = test[['Id','winPlacePerc']]
    
subm.to_csv("LGB.csv", index=False)
#=========================== xgboost ===================================

#----------------- 1 ------------------ 

import xgboost as xgb 

dtrain = xgb.DMatrix(df_train, label=Y)
dtest = xgb.DMatrix(df_test)

params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.05
         }
num_rounds = 50

xb = xgb.train(params, dtrain, num_rounds)

y_pred_xgb = xb.predict(dtest)

y_pred_xgb[y_pred_xgb > 1] = 1
    
test  = pd.read_csv('../input/test.csv')
test['winPlacePercPred'] = y_pred_xgb
aux = test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
    
subm = test[['Id','winPlacePerc']]
    
subm.to_csv("XGB1.csv", index=False)
#----------------- 2 ------------------ Score: 

import xgboost as xgb

dtrain = xgb.DMatrix(df_train, label = Y)
dtest = xgb.DMatrix(df_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(df_train, Y)

xgb_preds = model_xgb.predict(df_test)

xgb_preds[xgb_preds > 1] = 1
    
test  = pd.read_csv('../input/test.csv')
test['winPlacePercPred'] = xgb_preds
aux = test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
    
subm = test[['Id','winPlacePerc']]
    
subm.to_csv("XGB2.csv", index=False)
sub = xgb_preds * 0.1 + y_pred_xgb * 0.65 + lgb_pred * 0.25 # Score:

sub[sub > 1] = 1
    
test  = pd.read_csv('../input/test.csv')
test['winPlacePercPred'] = sub
aux = test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
test = test.merge(aux, how='left', on=['matchId','groupId'])
    
subm = test[['Id','winPlacePerc']]
    
subm.to_csv("Stacked_1.csv", index=False)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

LR = LinearRegression()

LR.fit(df_train, Y)
X_train, X_val, y_train,y_val = train_test_split(df_train,Y,test_size=0.3, random_state=42) 

print('Accuracy on training：\n',LR.score(X_train, y_train)) 
print('Accuracy on validation：\n',LR.score(X_val, y_val))
print('LinearRegression Accuracy：\n',LR.score(df_train, Y))

pred = LR.predict(df_test)
  
pred = pd.DataFrame({'Id':sum_id, 'winPlacePerc':pred}) 

pred.to_csv('pred_Linear.csv',index=None) 