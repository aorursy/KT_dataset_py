import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
os.getcwd()
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
hist=pd.read_csv('../input/historical_transactions.csv')
new=pd.read_csv('../input/new_merchant_transactions.csv')
#查看hist表中的category_1这个变量的类型有哪些，以便查看数据却缺失值及分布
hist['category_1'].describe()
hist['category_2'].dtype
hist['category_2'].describe()#有缺失值
hist['category_3'].describe()
hist['category_1'].value_counts()
temp=hist['category_2'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()
#缺失值处理
for df in [hist,new]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A', inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a', inplace=True)
#特征值探索
#分类变量处理
hist.info()
hist = pd.get_dummies(hist, columns=['category_2', 'category_3'])
new=pd.get_dummies(new,columns=['category_2', 'category_3'])
def binarize(df):
    for col in ['category_1','authorized_flag']:
        df[col]=df[col].map({'Y':1,'N':0})
    return df
hist=binarize(hist)
new=binarize(new)
def aggregate_data(df,prefix):
    df['purchase_date']=pd.DatetimeIndex(df['purchase_date']).astype(np.int64) * 1e-9
    agg_fun={'authorized_flag':['sum','mean'],
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
             'purchase_date': [np.ptp],
             'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
             'installments': ['sum', 'mean', 'max', 'min', 'std'],
             'month_lag': ['mean','min','max']
             }
    agg_trans=df.groupby('card_id').agg(agg_fun)
    agg_trans.columns=[prefix+'_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    return agg_trans
hist1=aggregate_data(hist,'hist_')
new1=aggregate_data(new,'new_')
# hist['card_id'].dtype
# hist['category_3_A'].dtype
#生成训练集和测试集
train['card_id']=train['card_id'].astype(str)
hist1['card_id']=hist1['card_id'].astype(str)
traindata=pd.merge(train,hist1,on='card_id',how='left')
traindata=pd.merge(traindata,new1,on='card_id',how='left')
testdata=pd.merge(test,hist1,on='card_id',how='left')
testdata=pd.merge(testdata,new1,on='card_id',how='left')
# traindata['hist_authorized_flag_sum'].sort_values()
# traindata['hist_authorized_flag_sum'].describe()
# import numpy as np
# from sklearn.model_selection import KFold
# X = ["a", "b", "c", "d"]
# kf = KFold(n_splits=2)
# for train, test in kf.split(X):
#      print("%s %s" % (train, test))
os.getcwd()
# testdata.to_csv('test.csv')
# target=traindata['target']
# traindata.to_csv('train.csv')
# del train['target']
# test.head()
# traindata.head()
# train.head()
train=traindata
test=testdata
target=train['target']
del train['target']
features = [c for c in traindata.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = ['feature_2', 'feature_3']
param = {'objective':'regression',
         'num_leaves': 31,
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
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission['target'] = predictions
sample_submission.to_csv('submission.csv', index=False)
