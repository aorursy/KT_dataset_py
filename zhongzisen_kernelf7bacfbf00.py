import numpy as mp
import pandas as pd
def memory_reduce(df):
    start_memory = df.memory_usage().sum()/1024**2
#     print("start_memory is {:.2f}".format(start_memory))
    for col in df.columns:
        col_type = df[col].dtype
        if(col_type != object):
            min_val = min(df[col])
            max_val = max(df[col])
            if(str(col_type)[:3] == 'int'):
                if(min_val > np.iinfo(np.int8).min and max_val < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif(min_val > np.iinfo(np.int16).min and max_val < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif(min_val > np.iinfo(np.int32).min and max_val < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif(min_val > np.iinfo(np.int64).min and max_val < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if(min_val > np.finfo(np.float16).min and max_val < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif(min_val > np.finfo(np.float32).min and max_val < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
#         else:
#             print(col)
#             df[col] = df[col].astype('category')
    end_memory = df.memory_usage().sum()/1024**2
#     print('end_memory is {:.2f}'.format(end_memory))
#     print((start_memory - end_memory) / start_memory)
    return df
# import pandas as pd
# import numpy as np
# train = pd.read_csv('../input/train_V2.csv' , nrows = 1000000)
# test = pd.read_csv('../input/test_V2.csv' , nrows = 1000000)
# # print(train['winPlacePerc'] * train['numGroups'])
# print((test['maxPlace'] == 1).sum())
# print((test['maxPlace'] == 0).sum())
# # print(train[['winPoints','winPlacePerc' , 'maxPlace' , 'rankPoints']])
import time
import numpy as np
import pandas as pd
import gc
startTime = time.time()
def feature_engineering(is_train = True , debug = True):
    test_Idx = None
    if(is_train):
        print('processing train data')
        if(debug):
            df = memory_reduce(pd.read_csv('../input/train_V2.csv' , nrows=10000))
        else:
            df = memory_reduce(pd.read_csv('../input/train_V2.csv'))
            df = df[pd.notnull(df['winPlacePerc'])]
    else:
        print('processing test data')
        if(debug):  
            df = memory_reduce(pd.read_csv('../input/test_V2.csv' , nrows = 10000))
        else:
            df = memory_reduce(pd.read_csv('../input/test_V2.csv'))
        test_Idx = df.Id
    print('remove many feature')
    target = 'winPlacePerc'
    features = list(df.columns)
    features.remove('Id')
    features.remove('groupId')
    features.remove('matchId')
    features.remove('matchType')
    y = None
    if(is_train):
        print('get target')
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean') , dtype=np.float64)
        features.remove(target)
        
    print('get group mean featuers')
    agg = df.groupby(['matchId' , 'groupId'])[features].agg('mean')
    agg_rank = agg.groupby(['matchId'])[features].rank(pct = True).reset_index()
    if(is_train):
        df_out = agg.reset_index()[['matchId','groupId']]
    else:
        df_out = df[['matchId' , 'groupId']]
        
    df_out = df_out.merge(agg.reset_index() , suffixes = ['',''] , how = 'left' , on = ['matchId' , 'groupId'])
    df_out = df_out.merge(agg_rank , suffixes = ['_mean','_mean_rank'] , how = 'left' , on = ['matchId' , 'groupId'])
    
    
    
    print('get group max features')
    agg = df.groupby(['matchId' , 'groupId'])[features].agg('max')
    agg_rank = agg.groupby(['matchId'])[features].rank(pct = True).reset_index()
    df_out = df_out.merge(agg.reset_index() , suffixes = ['',''] , how = 'left' , on = ['matchId' , 'groupId'])
    df_out = df_out.merge(agg_rank , suffixes = ['_max','_max_rank'] , how = 'left' , on = ['matchId' , 'groupId'])
    
    
    print('get group min features')
    agg = df.groupby(['matchId' , 'groupId'])[features].agg('min')
    agg_rank = agg.groupby(['matchId'])[features].rank(pct = True).reset_index()
    df_out = df_out.merge(agg.reset_index() , suffixes = ['',''] , how = 'left' , on = ['matchId' , 'groupId'])
    df_out = df_out.merge(agg_rank , suffixes = ['_min','_min_rank'] , how = 'left' , on = ['matchId' , 'groupId'])
    
    print('get group size feature')
    agg = df.groupby(['matchId','groupId']).size().reset_index(name = 'group_size')
    df_out = df_out.merge(agg , how = 'left' , on = ['matchId','groupId'])
    
    
    
    print('get group matchId mean')
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg , suffixes = ['','_match_mean'], how = 'left' , on = 'matchId')
    
    print('get match size feature')
    agg = df.groupby(['matchId']).size().reset_index(name = 'matchSize')
    df_out = df_out.merge(agg , on = 'matchId' , how = 'left')
    
    df_out.drop(['matchId' , 'groupId'] , axis = 1 ,inplace = True)
#     df_out = df_out.merge(df[['matchType']] , left_index = True , right_index = True)
    X = df_out
    columnsName = list(df_out.columns)
    del df , df_out , agg , agg_rank
    
    gc.collect()
    return X , y , columnsName , test_Idx
x_train , y_train  , train_columns , _ = feature_engineering(True , False)
x_test , _ , _ , test_Idx = feature_engineering(False , False)
print('This time is {:.2f}' .format(time.time() - startTime))
import time

# 移动总距离
x_train['totalDistance'] = x_train['swimDistance'] + x_train['rideDistance'] + x_train['walkDistance']
x_test['totalDistance'] = x_test['swimDistance'] + x_test['rideDistance'] + x_test['walkDistance']

# 药品总用量
x_train['healthItem'] = x_train['heals'] + x_train['boosts']
x_test['healthItem'] = x_test['heals'] + x_test['boosts']

# 爆头率
x_train['headshotRate'] =  x_train['kills'] / x_train['headshotKills']
x_test['headshotRate'] = x_test['kills'] / x_test['headshotKills']


# 短时间内杀人数占总杀人数比例
x_train['killStreakRate'] = x_train['killStreaks']/x_train['kills']
x_test['killStreakRate'] = x_test['killStreaks']/x_test['kills']


# 每分钟杀人数
x_train['killMinute'] = x_train['kills'] / x_train['matchDuration']
x_test['killMinute'] = x_test['kills'] / x_test['matchDuration']

# 每分钟造成伤害
x_train['damageDealtMinute'] = x_train['damageDealt'] / x_train['matchDuration']
x_test['damageDealtMinute'] = x_test['damageDealt'] / x_test['matchDuration']

# 总共参与的击杀人数
x_train['participateKills'] = x_train['kills'] + x_train['assists'] + x_train['DBNOs']
x_test['participateKills'] = x_test['kills'] + x_test['assists'] + x_test['DBNOs']

# 每分钟摧毁车辆数
x_train['vehicleDestroysMinute'] = x_train['vehicleDestroys'] / x_train['matchDuration']
x_test['vehicleDestroysMinute'] = x_test['vehicleDestroys'] / x_test['matchDuration']

# 在车上每米杀人数
x_train['killsMiter'] = x_train['roadKills'] / x_train['rideDistance']
x_test['killsMiter'] = x_test['roadKills'] / x_test['rideDistance']



del x_train['heals']
del x_test['heals']

train_columns.append('totalDistance')
train_columns.append('healthItems')
train_columns.append('headshotRate')
train_columns.append('killStreakRate')
train_columns.append('killMinute')
train_columns.append('damageDealtMinute')
train_columns.append('participateKills')
train_columns.append('vehicleDestroysMinute')
train_columns.append('killsMiter')
train_columns.remove('heals')

print(x_train.shape)
print('This time is {:.2f}'.format(time.time() - startTime))
import time
startTime = time.time()
x_train = memory_reduce(x_train)
x_test = memory_reduce(x_test)
print(time.time() - startTime)
# import lightgbm as lgb
# from sklearn.model_selection import KFold
# from lightgbm.sklearn import LGBMRegressor
# import pandas as pd
# import numpy as np
# import time
# import warnings
# warnings.filterwarnings('ignore')
# startTime = time.time()
# folds = KFold(n_splits = 3 , random_state = 6)
# oof_preds = np.zeros(x_train.shape[0])
# sub_preds = np.zeros(x_test.shape[0])
# valid_score = 0
# feature_importance_df = pd.DataFrame()
# for nfold , (trn_idx , val_idx) in enumerate(folds.split(x_train , y_train)):
#     trn_x , trn_y = x_train.iloc[trn_idx] , y_train[trn_idx]
    
#     val_x , val_y = x_train.iloc[val_idx] , y_train[val_idx]
    
    
#     train_data = lgb.Dataset(data = trn_x , label = trn_y)
#     valid_data = lgb.Dataset(data = val_x , label = val_y)
#     params = {'objective' : 'regression',
#               'metric' : 'mae',
#               'n_estimators' : 15000,
#               'early_stopping_rounds' : 100,
#               'num_leaves' : 31,
#               'learning_rate' : 0.05, 
#                "bagging_fraction" : 0.9,
#                "bagging_seed" : 0, 
#                "num_threads" : 4,
#                "colsample_bytree" : 0.7
#     }
#     lgb_model = lgb.train(params , train_data , valid_sets=[train_data , valid_data] , verbose_eval=100) 
#     lgb_pre = lgb_model.predict(val_x , num_iteration=lgb_model.best_iteration)
#     oof_preds[val_idx] = lgb_pre
#     oof_preds[oof_preds > 1] = 1
#     oof_preds[oof_preds < 0] = 0
#     sub_pred = lgb_model.predict(x_test , num_iteration=lgb_model.best_iteration)
#     sub_pred[sub_pred > 1] = 1
#     sub_pred[sub_pred < 0] = 0
#     sub_preds += sub_pred * 1.0 / folds.n_splits
#     fold_important_df = pd.DataFrame()
#     fold_important_df['feature'] = train_columns
#     fold_important_df['importance'] = lgb_model.feature_importance()
#     fold_important_df['fold'] = nfold + 1
#     feature_importance_df = pd.concat([feature_importance_df , fold_important_df] , axis = 0)
    
#     gc.collect()
    
    
    
#     print('The best_iteration {}'.format(lgb_model.best_iteration))

    
    
# print(time.time() - startTime)
import lightgbm as lgb
from sklearn.model_selection import KFold
from lightgbm.sklearn import LGBMRegressor
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
startTime = time.time()
print(startTime)
train_index = round(int(x_train.shape[0]*0.8))
dev_X = x_train[:train_index] 
val_X = x_train[train_index:]
dev_y = y_train[:train_index] 
val_y = y_train[train_index:] 
gc.collect();

# custom function to run light gbm model
def run_lgb(train_X, train_y, val_X, val_y, x_test):
    params = {"objective" : "regression", "metric" : "mae", 'n_estimators':20000, 'early_stopping_rounds':200,
              "num_leaves" : 31, "learning_rate" : 0.05, "bagging_fraction" : 0.7,
               "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7
             }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)
    
    pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)
    return pred_test_y, model

# Training the model #
pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, x_test)

print(time.time() - startTime)
# import seaborn as sns
# import matplotlib.pyplot as plt
# import time
# startTime = time.time()
# cols = fold_important_df[['feature','importance']].groupby('feature').agg('mean').sort_values('importance',ascending = False)[:50]
# fig , ax = plt.subplots(figsize = (15,15))
# sns.barplot(x = cols.importance , y = cols.index)
# print(time.time() - startTime)
import time
startTime = time.time()
df_sub = pd.read_csv("../input/sample_submission_V2.csv")
df_test = pd.read_csv("../input/test_V2.csv")
df_sub['winPlacePerc'] = pred_test
# Restore some columns
df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

# Sort, rank, and assign adjusted ratio
df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
df_sub_group = df_sub_group.merge(
    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
df_sub["winPlacePerc"] = df_sub["adjusted_perc"]

# Deal with edge cases
df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1

# Align with maxPlace
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
subset = df_sub.loc[df_sub.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case
df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
assert df_sub["winPlacePerc"].isnull().sum() == 0

# print(df_sub[["Id", "winPlacePerc"]])

df_sub[["Id", "winPlacePerc"]].to_csv("submission.csv", index=False)
print(time.time() - startTime)
# import time
# startTime = time.time()
# df_test = pd.read_csv('../input/' + 'test_V2.csv')
# pred = sub_preds
# for i in range(len(df_test)):
#     winPlacePerc = pred[i]
#     maxPlace = int(df_test.loc[i]['maxPlace'])
#     if(maxPlace == 0):
#         winPlacePerc = 0.0
#     elif(maxPlace == 1):
#         winPlacePerc = 1.0
#     else:
#         gap = 1.0 / (maxPlace - 1)
#         winPlacePerc = round(winPlacePerc / gap) * gap
#     if winPlacePerc < 0: winPlacePerc = 0.0
#     if winPlacePerc > 1: winPlacePerc = 1.0    
#     pred[i] = winPlacePerc

#     if (i + 1) % 100000 == 0:
#         print(i, flush=True, end=" ")

# df_test['winPlacePerc'] = pred
# submission = df_test[['Id' , 'winPlacePerc']]
# submission.to_csv('submission.csv' , index = False)
# print(time.time() - startTime)
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn
# import time
# from sklearn.preprocessing import LabelEncoder
# start = time.time()
# # 处理train中killPlace一个异常值
# train = train[train.killPlace <= 100]
# # 将rideDistance、walkDistance 、swimDistance相加求得移动总距离
# train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
# test['totalDistance'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']
# train['totalDistance']  = train['totalDistance'].apply(lambda dist : dist if dist >= 10 else 0)
# test['totalDistance']  = test['totalDistance'].apply(lambda dist : dist if dist >= 10 else 0)
# # 添加其他方式打死敌人数量
# train['otherKill'] = train['kills'] - (train['headshotKills'] + train['roadKills']) 
# test['otherKill'] = test['kills'] - (test['headshotKills'] + test['roadKills']) 
# # 删除游戏时长低于50秒的场次
# train = train[train.matchDuration > 50]
# test = test[test.matchDuration > 50]
# # 将杀人超过20的玩家改为20
# train['kills']  = train['kills'].apply(lambda kill : kill if kill <= 20 else 20)
# test['kills']  = test['kills'].apply(lambda kill : kill if kill <= 20 else 20)
# # 将杀死队友数量限制在0-5，其实最大只能杀死四个队友
# train['teamKills']  = train['teamKills'].apply(lambda teamKill : teamKill if teamKill <= 4 else 4)
# test['teamKills']  = test['teamKills'].apply(lambda teamKill : teamKill if teamKill <= 4 else 4)

# # 控制爆头数量
# train['headshotKills']  = train['headshotKills'].apply(lambda headshotKill : headshotKill if headshotKill <= 7 else 8)
# test['headshotKills']  = test['headshotKills'].apply(lambda headshotKill : headshotKill if headshotKill <= 7 else 8)

# # 对boosts数量控制
# train['boosts']  = train['boosts'].apply(lambda x : x if x <= 10 else 10)
# test['boosts']  = test['boosts'].apply(lambda x : x if x <= 10 else 10)

# # # 对使用药品数量进行合并 
# train['heals']  = train['heals'].apply(lambda heal : heal if heal <= 10 else 11)
# test['heals']  = test['heals'].apply(lambda heal : heal if heal <= 10 else 11)

# # 添加每分钟击杀敌人数量
# train['killMinute'] = train['kills'] / (train['matchDuration'] / 60)
# test['killMinute'] = test['kills'] / (test['matchDuration'] / 60)
# # 添加每分钟造成的伤害量
# train['damageMinute'] = train['damageDealt'] / (train['matchDuration'] / 60)
# test['damageMinute'] = test['damageDealt'] / (test['matchDuration'] / 60)
# # 对助攻数量进行合并
# train['assists']  = train['assists'].apply(lambda assist : assist if assist <= 14 else 14)
# test['assists']  = test['assists'].apply(lambda assist : assist if assist <= 14 else 14)
# # 对damageDealt正态化处理
# train['damageDealt'] = np.log1p(train['damageDealt'])



# # 添加是第一视角还是第三视角
# fpp_tpp = {'squad-fpp' : 'fpp',  
#                'duo' : 'tpp',
#                'solo-fpp' : 'fpp',
#                'squad' : 'tpp',
#                'duo-fpp' : 'fpp',
#                'solo' : 'tpp',
#                'normal-squad-fpp' : 'fpp',
#                'normal-solo-fpp' : 'fpp',
#                'normal-duo-fpp' : 'fpp',
#                'normal-duo' : 'tpp',
#                'normal-squad' : 'tpp',
#                'normal-solo' : 'tpp',
#                'crashfpp' : 'fpp',
#                'flaretpp' : 'tpp',
#                'flarefpp' : 'fpp',
#                'crashtpp' : 'tpp'}
# train['fpp_tpp'] = train['matchType'].replace(fpp_tpp)
# test['fpp_tpp'] = test['matchType'].replace(fpp_tpp)
# le = LabelEncoder().fit(['fpp','tpp'])
# train['fpp_tpp'] = le.transform(train['fpp_tpp'])
# test['fpp_tpp'] = le.transform(test['fpp_tpp'])
# # 添加游戏模式
# mode = {'squad-fpp' : 'nothing',  
#                'duo' : 'nothing',
#                'solo-fpp' : 'nothing',
#                'squad' : 'nothing',
#                'duo-fpp' : 'nothing',
#                'solo' : 'nothing',
#                'normal-squad-fpp' : 'normal',
#                'normal-solo-fpp' : 'normal',
#                'normal-duo-fpp' : 'normal',
#                'normal-duo' : 'normal',
#                'normal-squad' : 'normal',
#                'normal-solo' : 'normal',
#                'crashfpp' : 'crash',
#                'flaretpp' : 'flare',
#                'flarefpp' : 'flare',
#                'crashtpp' : 'crash'}
# train['playerMode'] = train['matchType'].replace(mode)
# test['playerMode'] = test['matchType'].replace(mode)
# le = LabelEncoder().fit(['nothing' , 'normal' , 'crash' , 'flare'])
# train['playerMode'] = le.transform(train['playerMode'])
# test['playerMode'] = le.transform(test['playerMode'])
# # 填充winPlacePerc的缺失值
# train['winPlacePerc'] = train['winPlacePerc'].fillna(train['winPlacePerc'].mean())
# label = train['winPlacePerc']



# del train['winPlacePerc']


# # 删除某些特征
# del train['Id']
# del train['groupId']
# del train['matchId']
# del train['matchType']

# del test['Id']
# del test['groupId']
# del test['matchId']
# del test['matchType']

# print(time.time() - start)
# import time
# startTime = time.time()
# # 求出一队中击杀敌人最多的数量      
# maxKills = train[['groupId','kills']].groupby(['groupId']).max()
# maxKills.rename(columns=lambda x : 'maxKills' , inplace=True)
# train = pd.merge(left = train , right = maxKills , left_on = 'groupId' , right_index=True)
# maxKills = test[['groupId','kills']].groupby(['groupId']).max()
# maxKills.rename(columns=lambda x : 'maxKills' , inplace=True)
# test = pd.merge(left = test , right = maxKills , left_on = 'groupId' , right_index=True)
# print(train.columns)
# print(time.time() - startTime)
# import time
# startTime = time.time()
# # 添加每队总计杀敌人数
# totalKills = train[['groupId','kills']].groupby(['groupId']).sum()
# totalKills.rename(columns=lambda x : 'totalKills' , inplace=True)
# train = pd.merge(left = train , right = totalKills , left_on = 'groupId' , right_index=True)
# totalKills = test[['groupId','kills']].groupby(['groupId']).sum()
# totalKills.rename(columns=lambda x : 'totalKills' , inplace=True)
# test = pd.merge(left = test , right = totalKills , left_on = 'groupId' , right_index=True)
# print(train.columns)
# print(time.time() - startTime)
# import time
# startTime = time.time()
# # 添加组内有几个人列
# size = train.groupby('groupId').size()
# size = size.to_frame()
# size.rename(columns = lambda x : 'groupSize' , inplace = True)
# train = pd.merge(left = train , right = size , left_on='groupId' , right_index=True)
# size = test.groupby('groupId').size()
# size = size.to_frame()
# size.rename(columns = lambda x : 'groupSize' , inplace = True)
# test = pd.merge(left = test , right = size , left_on='groupId' , right_index=True)
# print(train.columns)
# print(time.time() - startTime)
# import time
# startTime = time.time()
# # 添加每队平均杀敌人数
# # print(train[['groupSize_x','groupSize_y']])
# train['averageKills'] = round(train['totalKills'] / train['groupSize_x'])
# test['averageKills'] = round(test['totalKills'] / train['groupSize_x'])
# print(time.time() - startTime)
# # 将杀死队友数量限制在0-5，其实最大只能杀死四个队友
# train['weaponsAcquired']  = train['weaponsAcquired'].apply(lambda x : x if x <= 40 else 41)
# test['weaponsAcquired']  = test['weaponsAcquired'].apply(lambda x : x if x <= 40 else 41)
# # 将杀死队友数量限制在0-5，其实最大只能杀死四个队友
# train['boosts']  = train['boosts'].apply(lambda x : x if x <= 10 else 10)
# test['boosts']  = test['boosts'].apply(lambda x : x if x <= 10 else 10)
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# train_solo = train[train['playerGroupby'] == 'solo']
# train_duo = train[train['playerGroupby'] == 'duo']
# train_squad = train[train['playerGroupby'] == 'squad']
# train_random = train[train['playerGroupby'] == 'random']
# print((train_random['assists'] == 4).sum())
# print(train[(train['matchType'] == 'crashfpp')].groupby('numGroups').size())
# f , ax = plt.subplots(figsize = (15,15))
# sns.countplot(train['matchType'])
# plt.xticks(rotation = 90)

# f , ax = plt.subplots(figsize = (15,15))
# plt.pie(train.groupby(['matchType']).size() , labels=train.groupby(['matchType']).size().index)


# print((train['matchType'] == 'flashfpp').sum())
# train['k'] = train['DBNOs'] + train['kills']
# print(len(train) - len(train['Id'].unique()))
# print(train[['numGroups']])
# print(train[train['matchId'] == 'a10357fd1a4a91'].sort_values(['killPlace'])[['kills','killPlace','killPoints','winPlacePerc']])
# print(train[train['matchId'] == 'a10357fd1a4a91'].sort_values(['maxPlace']))

# print(train.sort_values('') [['maxPlace','numGroups']])

# corr = train.corr()
# corr = round(corr , 2)
# f, ax = plt.subplots(figsize=(15, 15))
# sns.heatmap(corr , annot=True , annot_kws={'size':9 , 'color':'black'})
# print(np.log1p(train['damageDealt']).skew())
# f, ax = plt.subplots(figsize=(12, 10))
# sns.distplot(train['matchType'])
# sns.distplot(train['totalDistance'])
# sns.distplot(np.log1p(train['totalDistance']))

# print(train['matchType'].unique())
# print((train['boosts'] == 20).sum())
# print(len(train[train['killPoints'] < 100]))
# print(min(train['killPoints'].unique()))
# print((train['boosts'] > 19).sum())
# print(train.groupby('boosts').size())


# trainRoad = train[train['matchType'] == 'normal-squad-fpp']
# trainCrashfpp = train[train['matchType'] == 'normal-squad-fpp']
# trainNotCrash = train[(train['matchType'] == 'squad-fpp') | (train['matchType'] == 'squad') | (train['matchType'] == 'duo-fpp') | (train['matchType'] == 'duo') | (train['matchType'] == 'solo-fpp') | (train['matchType'] == 'solo')]
# trainCrashtpp = train[(train['matchType'] == 'normal-squad-fpp') | (train['matchType'] == 'normal-squad') | (train['matchType'] == 'normal-duo-fpp') | (train['matchType'] == 'normal-duo') | (train['matchType'] == 'normal-solo-fpp') | (train['matchType'] == 'normal-solo')]
# print(len(trainCrash))
# f, ax = plt.subplots(figsize=(14, 14) , ncols=2 ,nrows=2)
# sns.boxplot(x=trainNotCrash['assists'] , y=trainNotCrash["winPlacePerc"] , ax=ax[0,0])
# sns.boxplot(x=trainCrashtpp['assists'] , y=trainCrashtpp["winPlacePerc"] , ax=ax[0,1])
# sns.boxplot(x=trainCrashfpp['revives'] , y=trainCrashfpp["winPlacePerc"] , ax=ax[1,0])
# sns.boxplot(x=train['assists'] , y=train["winPlacePerc"] , ax=ax[1,1])

# sns.scatterplot(x=trainNotCrash['matchDuration'] , y=trainNotCrash["winPlacePerc"] , ax=ax[0,0])
# sns.scatterplot(x=trainCrashtpp['matchDuration'] , y=trainCrashtpp["winPlacePerc"] , ax=ax[0,1])
# sns.scatterplot(x=trainCrashfpp['matchDuration'] , y=trainCrashfpp["winPlacePerc"] , ax=ax[1,0])
# sns.scatterplot(x=train['matchDuration'] , y=train["winPlacePerc"] , ax=ax[1,1])

# print((trainCrash['headshotKills'] == 2).sum())

# fig.axis(ymin=0, ymax=1)
# print(train['kills'].unique())
# print(len(train[train['assists']>=15]))
# print(train.groupby('assists').size())
# f , ax = plt.subplots(figsize = (20,20))
# f , ax = plt.subplots(ncols=2 , nrows=2, figsize = (20,20))
# sns.heatmap(train_solo.corr() , ax = ax[0,0] , cbar = False)
# sns.heatmap(train_duo.corr() , ax = ax[0,1] , cbar = False)
# sns.heatmap(train_squad.corr() , ax = ax[1,0] , cbar = False)
# sns.heatmap(train_random.corr() , ax = ax[1,1] , cbar = False)
# sns.boxplot(x = train_solo['assists'] , y = train_solo['winPlacePerc'] , ax=ax[0,0])
# sns.boxplot(x = train_duo['assists'] , y = train_duo['winPlacePerc'] , ax=ax[0,1])
# sns.boxplot(x = train_squad['assists'] , y = train_squad['winPlacePerc'] , ax=ax[1,0])
# print(train['Id'].head(10))
# train['firstName'] = train['groupId'].apply(lambda x : x[-1])
# sns.boxplot(x = train['playerGroupby'], y = train['winPlacePerc'])
# plt.show()
# 利用xgboost
# import numpy as np
# import pandas as pd
# from xgboost.sklearn import XGBRegressor
# import xgboost as xgb
# import time
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# startTime = time.time()
# train_X , test_X , train_Y , test_Y = train_test_split(train , label , test_size = 0.3)
# xgbRegressor = XGBRegressor(n_estimators = 500,
#                            max_depth = 6,
#                            learning_rate = 0.1,
#                            subsample = 0.5,
#                            colsample_bytree = 0.8,
#                            min_child_weight = 1)
# res = xgb.cv(params = xgbRegressor.get_xgb_params() , 
#             dtrain = xgb.DMatrix(train_X , train_Y),
#              num_boost_round=xgbRegressor.get_xgb_params()['n_estimators'],
#             early_stopping_rounds = 50)

# xgbRegressor = XGBRegressor(n_estimators = 600,
#                            max_depth = 6,
#                            learning_rate = 0.1,
#                            subsample = 0.8,
#                            colsample_bytree = 0.8,
#                            min_child_weight = 1)
# xgbRegressor.fit(train_X , train_Y)
# prediction = xgbRegressor.predict(test)

# sample = pd.read_csv('../input/sample_submission_V2.csv')
# sample['winPlacePerc'] = prediction
# sample.to_csv('sample_submission.csv',index = False)
# params = {"n_estimators" : [100,200]}
# grid = GridSearchCV(xgbRegressor , param_grid = params , cv = 3)
# grid = grid.fit(train_X , train_Y)
# print(grid.best_params_)
# print(res.shape[0])
# print(time.time() - startTime)


# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import warnings
# import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import RidgeCV
# from sklearn.ensemble import RandomForestRegressor
# import time
# warnings.filterwarnings("ignore")
# train_x , test_x, train_y , test_y = train_test_split(train , label , test_size = 0.3)

# # randomTree = RandomForestRegressor(max_features = 'auto'  , min_samples_leaf = 1  , min_samples_split = 4,
# #                                   n_estimators = 5)
# randomTree = RandomForestRegressor(max_features = 0.8 , n_estimators = 100)
# params = {'min_samples_leaf' : [1,3],
#          'min_samples_split' : [2,4]}
# grid = GridSearchCV(estimator=randomTree , param_grid=params)
# grid = grid.fit(train_x , train_y)
# print(grid.best_params_)
# randomTree = RandomForestRegressor(max_features = 0.8  
#                                    , min_samples_leaf = grid.best_params_['min_samples_leaf'] 
#                                    , min_samples_split = grid.best_params_['min_samples_split']
#                                    , n_estimators = grid.best_params_['n_estimators'])


# randomTree.fit(train , label)

# prediction = randomTree.predict(test)

# sample = pd.read_csv('../input/sample_submission_V2.csv')
# sample['winPlacePerc'] = prediction
# sample.to_csv('sample_submission.csv',index = False)
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn
# # 处理train中killPlace一个异常值
# train = train[train.killPlace <= 100]
# # 将rideDistance、walkDistance 、swimDistance相加求得移动总距离
# train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
# test['totalDistance'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']
# # 添加其他方式打死敌人数量
# train['otherKill'] = train['kills'] - (train['headshotKills'] + train['roadKills']) 
# test['otherKill'] = test['kills'] - (test['headshotKills'] + test['roadKills']) 
# # 删除游戏时长低于50秒的场次
# train = train[train.matchDuration > 50]
# test = test[test.matchDuration > 50]
# # 添加每分钟击杀敌人数量
# train['killMinute'] = train['kills'] / (train['matchDuration'] / 60)
# test['killMinute'] = test['kills'] / (test['matchDuration'] / 60)
# # 添加每分钟造成的伤害量
# train['damageMinute'] = train['damageDealt'] / (train['matchDuration'] / 60)
# test['damageMinute'] = test['damageDealt'] / (test['matchDuration'] / 60)
# # 添加组内有几个人列
# size = train.groupby('groupId').size()
# size = size.to_frame()
# size.rename(columns = lambda x : 'groupSize' , inplace = True)
# train = pd.merge(left = train , right = size , left_on='groupId' , right_index=True)
# size = test.groupby('groupId').size()
# size = size.to_frame()
# size.rename(columns = lambda x : 'groupSize' , inplace = True)
# test = pd.merge(left = test , right = size , left_on='groupId' , right_index=True)
# # 填充winPlacePerc的缺失值
# train['winPlacePerc'] = train['winPlacePerc'].fillna(train['winPlacePerc'].mean())



# # 提取Id第一个字母
# # train['IdFirst'] = train['Id'].apply(lambda x : x[0])
# # test['IdFirst'] = test['Id'].apply(lambda x : x[0])



# print(len(train.columns))
# print(len(test.columns))
# print('stop')
        # 对同一队的数据进行分析

# 求出一队中击杀敌人最多的数量      
# maxKills = train[['groupId','kills']].groupby(['groupId']).max()
# maxKills.rename(columns=lambda x : 'maxKills' , inplace=True)
# train = pd.merge(left = train , right = maxKills , left_on = 'groupId' , right_index=True)
# maxKills = test[['groupId','kills']].groupby(['groupId']).max()
# maxKills.rename(columns=lambda x : 'maxKills' , inplace=True)
# test = pd.merge(left = test , right = maxKills , left_on = 'groupId' , right_index=True)
# print(train.head())
# 添加每队总计杀敌人数
# totalKills = train[['groupId','kills']].groupby(['groupId']).sum()
# totalKills.rename(columns=lambda x : 'totalKills' , inplace=True)
# train = pd.merge(left = train , right = totalKills , left_on = 'groupId' , right_index=True)
# totalKills = test[['groupId','kills']].groupby(['groupId']).sum()
# totalKills.rename(columns=lambda x : 'totalKills' , inplace=True)
# test = pd.merge(left = test , right = totalKills , left_on = 'groupId' , right_index=True)
# print(train.head())
# 添加每队平均杀敌人数
# train['averageKills'] = train['totalKills'] / train['groupSize']
# test['averageKills'] = test['totalKills'] / train['groupSize']
# print(train.head())
# 一队中造成伤害最多
# maxDamageDealt = train[['groupId','damageDealt']].groupby(['groupId']).max()
# maxDamageDealt.rename(columns=lambda x : 'maxDamageDealt' , inplace=True)
# train = pd.merge(left = train , right = maxDamageDealt , left_on = 'groupId' , right_index=True)
# maxDamageDealt = test[['groupId','damageDealt']].groupby(['groupId']).max()
# maxDamageDealt.rename(columns=lambda x : 'maxKills' , inplace=True)
# test = pd.merge(left = test , right = maxDamageDealt , left_on = 'groupId' , right_index=True)
# print(train.head())
# 添加每队总伤害
# totalDamageDealt = train[['groupId','damageDealt']].groupby(['groupId']).sum()
# totalDamageDealt.rename(columns=lambda x : 'totalDamageDealt' , inplace=True)
# train = pd.merge(left = train , right = totalDamageDealt , left_on = 'groupId' , right_index=True)
# totalDamageDealt = test[['groupId','damageDealt']].groupby(['groupId']).sum()
# totalDamageDealt.rename(columns=lambda x : 'totalDamageDealt' , inplace=True)
# test = pd.merge(left = test , right = totalDamageDealt , left_on = 'groupId' , right_index=True)
# print(train.head())
# 添加每队平均伤害
# train['averageDamageDealt'] = train['totalDamageDealt'] / train['groupSize']
# test['averageDamageDealt'] = test['totalDamageDealt'] / train['groupSize']

# print(train.head())
# 添加全队一共行进距离
# groupDistance = train[['groupId','totalDistance']].groupby(['groupId']).sum()
# groupDistance.rename(columns=lambda x : 'groupDistance' , inplace=True)
# train = pd.merge(left = train , right = groupDistance , left_on = 'groupId' , right_index=True)
# groupDistance = test[['groupId','totalDistance']].groupby(['groupId']).sum()
# groupDistance.rename(columns=lambda x : 'groupDistance' , inplace=True)
# test = pd.merge(left = test , right = groupDistance , left_on = 'groupId' , right_index=True)
# print(train.head())
# 添加全队行进的最远距离
# maxDistance = train[['groupId','totalDistance']].groupby(['groupId']).max()
# maxDistance.rename(columns=lambda x : 'maxDistance' , inplace=True)
# train = pd.merge(left = train , right = maxDistance , left_on = 'groupId' , right_index=True)
# maxDistance = test[['groupId','totalDistance']].groupby(['groupId']).max()
# maxDistance.rename(columns=lambda x : 'maxDistance' , inplace=True)
# test = pd.merge(left = test , right = maxDistance , left_on = 'groupId' , right_index=True)
# print(train.head())
# 添加全队行进的最近距离
# minDistance = train[['groupId','totalDistance']].groupby(['groupId']).min()
# minDistance.rename(columns=lambda x : 'minDistance' , inplace=True)
# train = pd.merge(left = train , right = minDistance , left_on = 'groupId' , right_index=True)
# minDistance = test[['groupId','totalDistance']].groupby(['groupId']).min()
# minDistance.rename(columns=lambda x : 'minDistance' , inplace=True)
# test = pd.merge(left = test , right = minDistance , left_on = 'groupId' , right_index=True)
# print(train.head())
# 添加全队平均行进距离
# train['meanDistance'] = train['totalDistance'] / train['groupSize']
# test['meanDistance'] = test['totalDistance'] / train['groupSize']

# print(train.head())
# 助攻总数
# sumAssists = train[['groupId','assists']].groupby(['groupId']).sum()
# sumAssists.rename(columns=lambda x : 'sumAssists' , inplace=True)
# train = pd.merge(left = train , right = sumAssists , left_on = 'groupId' , right_index=True)
# sumAssists = test[['groupId','assists']].groupby(['groupId']).sum()
# sumAssists.rename(columns=lambda x : 'sumAssists' , inplace=True)
# test = pd.merge(left = test , right = sumAssists , left_on = 'groupId' , right_index=True)
# print(train.head())
# 助攻最大值
# maxAssists = train[['groupId','assists']].groupby(['groupId']).max()
# maxAssists.rename(columns=lambda x : 'maxAssists' , inplace=True)
# train = pd.merge(left = train , right = maxAssists , left_on = 'groupId' , right_index=True)
# maxAssists = test[['groupId','assists']].groupby(['groupId']).max()
# maxAssists.rename(columns=lambda x : 'maxAssists' , inplace=True)
# test = pd.merge(left = test , right = maxAssists , left_on = 'groupId' , right_index=True)
# print(train.head())
# 助攻平均数
# train['meanAssists'] = train['sumAssists'] / train['groupSize']
# test['meanAssists'] = test['sumAssists'] / train['groupSize']
# print(train.head())
# 击倒总数
# sumDBNOs = train[['groupId','DBNOs']].groupby(['groupId']).sum()
# sumDBNOs.rename(columns=lambda x : 'sumDBNOs' , inplace=True)
# train = pd.merge(left = train , right = sumDBNOs , left_on = 'groupId' , right_index=True)
# sumDBNOs = test[['groupId','DBNOs']].groupby(['groupId']).sum()
# sumDBNOs.rename(columns=lambda x : 'sumDBNOs' , inplace=True)
# test = pd.merge(left = test , right = sumDBNOs , left_on = 'groupId' , right_index=True)
# print(train.head())
# 击倒平均数
# train['meanDBNOs'] = train['sumDBNOs'] / train['groupSize']
# test['meanDBNOs'] = test['sumDBNOs'] / train['groupSize']
# print(train.head())
# 最大击倒数
# maxDBNOs = train[['groupId','DBNOs']].groupby(['groupId']).max()
# maxDBNOs.rename(columns=lambda x : 'maxDBNOs' , inplace=True)
# train = pd.merge(left = train , right = maxDBNOs , left_on = 'groupId' , right_index=True)
# maxDBNOs = test[['groupId','DBNOs']].groupby(['groupId']).max()
# maxDBNOs.rename(columns=lambda x : 'maxDBNOs' , inplace=True)
# test = pd.merge(left = test , right = maxDBNOs , left_on = 'groupId' , right_index=True)
# print(train.head())
# 删除某些特征
# del train['Id']
# del train['matchId']
# del train['matchType']
# del train['DBNOs']
# del train['assists']
# del train['damageDealt']
# del train['heals']
# del train['kills']
# del train['totalDistance']


# del test['Id']
# del test['matchId']
# del test['matchType']
# del test['DBNOs']
# del test['assists']
# del test['damageDealt']
# del test['heals']
# del test['kills']
# del test['totalDistance']
# print(len(train.columns))
# print(len(test.columns))

# train.drop_duplicates(subset=['groupId'] , inplace = True)
# testFlag = test.drop_duplicates(subset=['groupId'])
# label = train['winPlacePerc']
# dropGroupId = testFlag['groupId'].to_frame()
# del train['winPlacePerc']
# del train['groupId']
# del testFlag['groupId']
# print(len(train.columns))
# print(len(testFlag.columns))
# print('stop')
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import warnings
# import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import RidgeCV
# from sklearn.ensemble import RandomForestRegressor
# import time
# warnings.filterwarnings("ignore")
# # train_x , test_x, train_y , test_y = train_test_split(train , label , test_size = 0.3)
# # 岭回归，效果奇差
# # ridge = RidgeCV(alphas=[0.01,0.03,0.09,0.3,1,3,10,30,60])
# # ridge.fit(train_x , train_y)
# # alpha = ridge.alpha_

# # ridge = RidgeCV(alphas = [alpha * 0.6 , alpha * 0.65 ,alpha * 0.7 ,alpha * 0.75 ,alpha * 0.8 ,
# #                                               alpha * 0.85 ,alpha * 0.9 ,alpha * 0.95 ,alpha * 1 ,
# #                                               alpha * 1.05 ,alpha * 1.1 ,alpha * 1.15 ,alpha * 1.2 ,alpha * 1.25 ])
# # # ridge = RidgeCV(alphas = 0.375)
# # ridge.fit(train_x , train_y)
# # alpha = ridge.alpha_
# # prediction = ridge.predict(test)
# # ti = time.time()
# # print(ti)

# randomTree = RandomForestRegressor(max_features = 0.7  , min_samples_leaf = 1  , min_samples_split = 4,
#                                   n_estimators = 20)
# # params = {'min_samples_leaf' : [1,3],
# #          'min_samples_split' : [2,4],
# #          'n_estimators' : [200,400]}
# # grid = GridSearchCV(estimator=randomTree , param_grid=params)
# # print(1)
# # grid = grid.fit(train , label)
# # print(2)
# # print(grid.best_score_)
# # print(grid.best_params_)
# randomTree.fit(train , label)

# prediction = randomTree.predict(testFlag)
# dropGroupId['winPlacePerc'] = prediction
# prediction = pd.merge(test , dropGroupId , left_on = 'groupId' , right_on = 'groupId')['winPlacePerc']
# sample = pd.read_csv('../input/sample_submission_V2.csv')
# sample['winPlacePerc'] = prediction
# sample.to_csv('sample_submission.csv',index = False)
# print('stop')
# # print(sample)
# print(np.sqrt(sklearn.metrics.mean_absolute_error(ridge.predict(test_x) , test_y)))



# train['normal'] = train['matchType'].apply(lambda x : 0 if (x.find('flare') >=0 or x.find('crash') >=0) else 1)
# train['normal'] = train['matchType'].apply(lambda x : 0 if (x.find('solo') >=0) else 1)
# print(train.groupby(['normal']).mean()['winPlacePerc'])


# trainNO = train[train['normal'] == 0]
# print(trainNO['winPlacePerc'].mean())
# print(train['winPlacePerc'].mean())
# print(trainNO[['matchType','revives','teamKills','assists']])


# print(train[['matchType','killPlace']])
# print(time.time() - ti)
# print('stop')
# plt.scatter(train.totalDistance , train.winPlacePerc)
# plt.show()
# train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']

# print(train['heals'].sort_values(ascending = False))
# f , ax = plt.subplots(figsize = (12,12))
# sns.countplot(x = 'kills' , data = test)
# plt.show()
# print(train['matchDuration'])