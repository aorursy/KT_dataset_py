import numpy as np
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from xgboost import plot_importance
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
hist=pd.read_csv('../input/historical_transactions.csv')
new=pd.read_csv('../input/new_merchant_transactions.csv')
#缺失值填充
for df in [hist,new]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A', inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
#日期处理及分类特征探索
hist = pd.get_dummies(hist, columns=['category_2', 'category_3'])
new=pd.get_dummies(new,columns=['category_2', 'category_3'])

for df in [hist,new]:
    df['purchase_date']=pd.to_datetime(df['purchase_date'])## 将交易日期由字符串改为时间变量
    df['year']=df['purchase_date'].dt.year
    df['month']=df['purchase_date'].dt.month
    df['weekofyear']=df['purchase_date'].dt.weekofyear#一年中的第几周
    df['dayofweek']=df['purchase_date'].dt.dayofweek#一周中的第几天
    df['weekend'] =(df['purchase_date'].dt.weekday>=5).astype(int)#星期一到星期日是0-6表示
    df['hour'] = df['purchase_date'].dt.hour
    df['category_1']=df['category_1'].map({'Y':1,'N':0})
    df['authorized_flag']=df['authorized_flag'].map({'Y':1, 'N':0})
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days) //30#取整
    
hist.describe()
#特征工程
def aggregate_data(df,prefix):
    df['purchase_date']=pd.DatetimeIndex(df['purchase_date']).astype(np.int64) * 1e-9
    agg_fun={'authorized_flag':['count','sum','mean'],
             'category_1':['sum','mean'],
             'category_2_1.0':['sum','mean'],
             'category_2_2.0': ['sum', 'mean'],
             'category_2_3.0': ['sum', 'mean'],
             'category_2_4.0': ['sum', 'mean'],
             'category_2_5.0': ['sum', 'mean'],
             'category_3_A': ['sum', 'mean'],
             'category_3_B': ['sum', 'mean'],
             'category_3_C': ['sum', 'mean'],
             'merchant_id': ['nunique'],
             'merchant_category_id':['nunique'],
             'state_id': ['nunique'],
             'city_id': ['nunique'],
             'purchase_date': ['min','max',np.ptp],
             'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
             'installments': ['sum', 'mean', 'max', 'min', 'std'],
             'month_lag': ['mean','min','max'],
             'subsector_id':['nunique'],
             'year':['nunique'],
             'month':['nunique'],
             'weekofyear':['nunique'],
             'weekend': ['sum','mean'],
             'hour': ['nunique'],
             'dayofweek': ['nunique'],
             'month_diff':['mean']
             }
    agg_trans=df.groupby('card_id').agg(agg_fun)
    agg_trans.columns=[prefix+'_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    return agg_trans
hist1=aggregate_data(hist,'hist_')
new1=aggregate_data(new,'new_')
hist1.info()
new1.info()
#生成训练集和测试集
train['card_id']=train['card_id'].astype(str)
hist1['card_id']=hist1['card_id'].astype(str)
traindata=pd.merge(train,hist1,on='card_id',how='left')
traindata=pd.merge(traindata,new1,on='card_id',how='left')
testdata=pd.merge(test,hist1,on='card_id',how='left')
testdata=pd.merge(testdata,new1,on='card_id',how='left')
traindata.head()
#最近一次交易和平均购买间隔
for df in [traindata,testdata]:
    df['hist_new_average']=(df['new_purchase_date_ptp']+df['hist_purchase_date_ptp'])/(df['new_authorized_flag_count']+df['hist_authorized_flag_count'])
    df['hist_average']=df['hist_purchase_date_ptp']/df['hist_authorized_flag_count']
    df['new_average']=df['new_purchase_date_ptp']/df['new_authorized_flag_count']
train=traindata
test=testdata
target=train['target']
del train['target']
features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = ['feature_2', 'feature_3']
#lgb
param = {'objective':'regression',
         'num_leaves': 80,
         'min_data_in_leaf': 25,
         'max_depth': 7,
         'learning_rate': 0.01,
         'lambda_l1':0.13,
         "boosting": "gbdt",
         "feature_fraction":0.85,
         'bagging_freq':8,
         "bagging_fraction": 0.9 ,
         "metric": 'rmse',
         "verbosity": -1,
         "random_state": 2333}
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()
train.head(5)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
                           categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][features],
                           label=target.iloc[val_idx],
                           categorical_feature=categorical_feats )
    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds = 200)
    
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,25))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",
                                           ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
# sample_submission = pd.read_csv('../input/sample_submission.csv')
# sample_submission['target'] = predictions
# sample_submission.to_csv('submission_lgb.csv', index=False)
##xgb model
xgb_params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'gamma': 0.1,
    'max_depth': 6,
    'eval_metric':'rmse',
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}
folds= KFold(n_splits=5, shuffle=True, random_state=15)
oof_xgb = np.zeros(len(train))
predictions_xgb = np.zeros(len(test))
for fold_,(trn_idx,val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = xgb.DMatrix(train.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
                           #categorical_feature=categorical_feats
                          )
    val_data = xgb.DMatrix(train.iloc[val_idx][features],
                           label=target.iloc[val_idx],
                           #categorical_feature=categorical_feats 
                          )
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
    print("xgb " + str(fold_) + "-" * 50)
    num_rounds = 2000
    xgb_model = xgb.train(xgb_params, trn_data, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=1000)
    oof_xgb[val_idx] = xgb_model.predict(xgb.DMatrix(train.iloc[val_idx][features]), ntree_limit=xgb_model.best_ntree_limit+50)
    predictions_xgb += xgb_model.predict(xgb.DMatrix(test[features]), ntree_limit=xgb_model.best_ntree_limit+50) / folds.n_splits
print("CV score: {:<8.5f}".format(mean_squared_error(target.values,oof_xgb)**0.5))
plot_importance(xgb_model)#
plt.show()
#融合
total_sum=0.5*predictions+0.5*predictions_xgb
sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df['target'] = total_sum
sub_df.to_csv('submission_lgbxgb.csv', index=False)