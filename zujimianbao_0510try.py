# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.metrics import roc_auc_score,auc
from  sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
pd.set_option("display.max.columns",None)




#获取数据
train_beh=pd.read_csv("../input/zsfintech/_beh.csv")
train_tag=pd.read_csv("../input/zsfintech/_tag.csv")
train_trd=pd.read_csv("../input/zsfintech/_trd.csv")
test_beh=pd.read_csv("../input/zsfintech/test_beh_b.csv")
test_tag=pd.read_csv("../input/zsfintech/test_tag_b.csv")
test_trd=pd.read_csv("../input/zsfintech/test_trd_b.csv")
train_tag.set_index('id', inplace=True)
test_tag.set_index('id', inplace=True)
train_beh.page_tm=train_beh['Unnamed: 3']
train_beh.drop('Unnamed: 3',axis=1,inplace=True)
test_beh.page_tm=test_beh['Unnamed: 2']
test_beh.drop('Unnamed: 2',axis=1,inplace=True)

train_trd=train_trd.sort_values(['id','trx_tm'])
train_trd['trx_tm_day'] = train_trd['trx_tm'].apply(lambda x:x.split(' ')[0])
train_trd['trx_tm_hour'] = train_trd['trx_tm'].apply(lambda x:x.split(' ')[1].split(':')[0])
test_trd=test_trd.sort_values(['id','trx_tm'])
test_trd['trx_tm_day'] = test_trd['trx_tm'].apply(lambda x:x.split(' ')[0])
test_trd['trx_tm_hour'] = test_trd['trx_tm'].apply(lambda x:x.split(' ')[1].split(':')[0])



train_beh=train_beh.sort_values(['id','page_tm'])
train_beh['page_tm_day'] = train_beh['page_tm'].apply(lambda x:x.split(' ')[0])
train_beh['page_tm_hour'] = train_beh['page_tm'].apply(lambda x:x.split(' ')[1].split(':')[0])
test_beh=test_beh.sort_values(['id','page_tm'])
test_beh['page_tm_day'] = test_beh['page_tm'].apply(lambda x:x.split(' ')[0])
test_beh['page_tm_hour'] = test_beh['page_tm'].apply(lambda x:x.split(' ')[1].split(':')[0])
train_trd

#trd、beh表中信息
def zj(df,trd,beh):
    df['click_times']=beh['id'].value_counts()
    df['click_times'].fillna(0,inplace=True)
    
#   for i in train_beh.page_tm_day.unique():
#        df['page_tm_day'+str(i)+'times']=beh[beh.page_tm_day==i].id.value_counts()
#        df['page_tm_day'+str(i)+'times'].fillna(0,inplace=True)     

#   for i in train_beh.page_tm_hour.unique():
#        df['page_tm_hour'+str(i)+'times']=beh[beh.page_tm_hour==i].id.value_counts()
#        df['page_tm_hour'+str(i)+'times'].fillna(0,inplace=True)   

    df['shouzhi']=trd.groupby(by=['id']).cny_trx_amt.sum()
    df['shouzhi'].fillna(0,inplace=True)
    
    df['shouzhi_times']=trd['id'].value_counts()
    df['shouzhi_times'].fillna(0,inplace=True)
    
    df['shouru']=trd[trd.cny_trx_amt>0].groupby(by=['id']).cny_trx_amt.sum()
    df['shouru'].fillna(0,inplace=True)
    
    df['shouru_times']=trd[trd.cny_trx_amt>0].groupby(by=['id']).cny_trx_amt.count()
    df['shouru_times'].fillna(0,inplace=True)
    
    df['maxshouru']=trd[trd.cny_trx_amt>0].groupby(by=['id']).cny_trx_amt.max()
    df['maxshouru'].fillna(0,inplace=True)
    
    df['maxzhichu']=trd[trd.cny_trx_amt<0].groupby(by=['id']).cny_trx_amt.min()
    df['maxzhichu'].fillna(0,inplace=True)
    
    df['shouzhi_mean']=trd.groupby(by=['id']).cny_trx_amt.mean()
    df['shouzhi_mean'].fillna(0,inplace=True)
    
    df['shouru_mean']=trd[trd.cny_trx_amt>0].groupby(by=['id']).cny_trx_amt.mean()
    df['shouru_mean'].fillna(0,inplace=True)   
    
    df['Dat_Flg1_Cd_btimes']=trd[trd.Dat_Flg1_Cd=='B'].id.value_counts()
    df['Dat_Flg1_Cd_btimes'].fillna(0,inplace=True)
    
    df['Dat_Flg3_Cd_atimes']=trd[trd.Dat_Flg3_Cd=='A'].id.value_counts()
    df['Dat_Flg3_Cd_atimes'].fillna(0,inplace=True)
    
    df['Dat_Flg3_Cd_btimes']=trd[trd.Dat_Flg3_Cd=='B'].id.value_counts()
    df['Dat_Flg3_Cd_btimes'].fillna(0,inplace=True)
    
    df['Trx_Cod1_Cd_1times']=trd[trd.Trx_Cod1_Cd==1].id.value_counts()
    df['Trx_Cod1_Cd_1times'].fillna(0,inplace=True)
    
    df['Trx_Cod1_Cd_2times']=trd[trd.Trx_Cod1_Cd==2].id.value_counts()
    df['Trx_Cod1_Cd_2times'].fillna(0,inplace=True)
    
    for i in train_trd.Trx_Cod2_Cd.unique():
        df['Trx_Cod2_Cd_'+str(i)+'times']=trd[trd.Trx_Cod2_Cd==i].id.value_counts()
        df['Trx_Cod2_Cd_'+str(i)+'times'].fillna(0,inplace=True)
    
    for i in train_trd.trx_tm_day.unique():
        df['trx_tm_day'+str(i)+'times']=trd[trd.trx_tm_day==i].id.value_counts()
        df['trx_tm_day'+str(i)+'times'].fillna(0,inplace=True)    
    
#   for i in train_trd.trx_tm_hour.unique():
#        df['trx_tm_hour'+str(i)+'times']=trd[trd.trx_tm_hour==i].id.value_counts()
#        df['trx_tm_hour'+str(i)+'times'].fillna(0,inplace=True)   
    
#    df['trd_effective_days']=trd.groupby(by=['id']).trx_tm_day.nunique()
#    df['trd_effective_days'].fillna(0,inplace=True)
    
    df.replace(r'\N',np.nan,inplace=True)
    df.replace('~',np.nan,inplace=True)
    df.replace('-1',np.nan,inplace=True)
    df.replace(-1,np.nan,inplace=True)
    return df
train_tag_zj=zj(train_tag,train_trd,train_beh)
test_tag_zj=zj(test_tag,test_trd,test_beh)
train_tag_zj
#调整一些以字母形式出现的数字
def zl(df):
    zmlist=['gdr_cd','mrg_situ_cd','acdm_deg_cd','deg_cd','edu_deg_cd','atdd_type']
    df_zl=df.drop(zmlist,axis=1).astype(float)
    dummies_acdm = pd.get_dummies(df.acdm_deg_cd, prefix= 'acdm_deg_cd')
    dummies_deg = pd.get_dummies(df.deg_cd, prefix= 'deg_cd')    
    df_zl.loc[df['gdr_cd']=='M','sex']=1
    df_zl.loc[df['gdr_cd']=='F','sex']=0
    df_zl.loc[df['mrg_situ_cd']=='Z','mrg']=2
    df_zl.loc[df['mrg_situ_cd']=='A','mrg']=1
    df_zl.mrg.fillna(0,inplace=True)    
    df_zl = pd.concat([df_zl,dummies_acdm,dummies_deg], axis=1)
    return df_zl
train_tag_zl=zl(train_tag_zj)
test_tag_zl=zl(test_tag_zj)
train_tag_zl
x=train_tag_zl.iloc[:,1:]
y=train_tag_zl.iloc[:,0]
x_test=test_tag_zl
#划分测试集
x_train,x_cv,y_train,y_cv=train_test_split(x,y,test_size=0.1,random_state=101)
x_train
train_tag_cgb=xgb.DMatrix(x,label=y)
xgb1 = XGBClassifier(learning_rate=0.01,
                      n_estimators=2086,
                      silent=False,
                      objective='binary:logistic',
                      booster='gbtree',
                      n_jobs=4,
                      gamma=7,
                      min_child_weight=4,
                      subsample=0.8,
                      colsample_bytree=0.8,
                      colsample_bylevel=0.8,
                      reg_lambda=1,
                      random_state=101)
#cv_result = xgb.cv(xgb1.get_xgb_params(), train_tag_cgb, num_boost_round=xgb1.get_xgb_params()['n_estimators'],
#                   nfold=5, metrics='auc',early_stopping_rounds=50,
#                   callbacks=[xgb.callback.print_evaluation(show_stdv=False),
#                              xgb.callback.early_stop(50) ])

#param_grid = {'min_child_weight':[1,2,3,4,5]}
#grid_search = GridSearchCV(xgb1,param_grid,scoring='roc_auc',iid=False,cv=5)
#grid_search.fit(x,y)
#print('best_params:',grid_search.best_params_)
#print('best_score:',grid_search.best_score_)
xgb2=xgb1.fit(x_train,y_train)
y_cv_pred=xgb2.predict_proba(x_cv)[:,1]
y_train_pred = xgb2.predict_proba(x_train)[:,1]
print(roc_auc_score(y_train,y_train_pred))
print(roc_auc_score(y_cv,y_cv_pred))
#featureim=pd.DataFrame({'feature':x_train.columns,'score':xgb1.feature_importances_})
#feature_sel=featureim[featureim.score>0]
#feature_sel.sort_values(by='score')
#x_train=x_train[feature_sel.feature]
#x_cv=x_cv[feature_sel.feature]
#x_test=x_test[feature_sel.feature]

#lgb3=lgb1.fit(x_train,y_train,categorical_feature=cat_news)
#y_cv_pred= lgb3.predict_proba(x_cv)[:,1]
#y_train_pred = lgb3.predict_proba(x_train)[:,1]
#print(roc_auc_score(y_train,y_train_pred))
#print(roc_auc_score(y_cv,y_cv_pred))
#输出到txt
xgb2=xgb1.fit(x,y)
y_pred = xgb2.predict_proba(x_test)[:,1]
out=pd.DataFrame({'id':x_test.index,'score':y_pred})
out.to_csv('05121.txt',sep='\t',header=False,index=False)