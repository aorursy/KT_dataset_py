# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.metrics import auc

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from datetime import datetime, date, time, timedelta



def label_var(data,variables_cat):

    lb=[]

    for m in variables_cat:

        l=LabelEncoder()

        lb.append(l.fit(list(data[m].dropna())))

    

    return lb



def label_enc(data,l,categorical_features):

    i=0

    for m in categorical_features:

        data.loc[data[m].notnull(),m]=l[i].transform(data.loc[data[m].notnull(),m])

        i=i+1

        



df_tr = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")

df_ts = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")
feature_importance_dfs = {}



for key, value in feature_importance_dfs.items():

    print('Feature Importance for type ', encoders['type'].classes_[key], ' :')

    print(

    value.groupby(['Feature'])[['importance']].mean().sort_values(

        "importance", ascending=False).head(20)

         )
"apache_3j_diagnosis" in df_tr.columns.tolist()

feature_importance_dfs = {}
grouped = df_tr.groupby(['apache_3j_diagnosis'])[

'd1_heartrate_min',

'apache_4a_hospital_death_prob',

'age',

'd1_temp_max',

'd1_platelets_min','bmi','d1_spo2_min',

'd1_heartrate_max',

'd1_creatinine_max',

'd1_wbc_min',

'urineoutput_apache',

'h1_heartrate_max',

'heart_rate_apache',

'd1_sodium_max',

'glucose_apache',

'd1_glucose_min',

'd1_lactate_min',

'weight',

'd1_bun_min',

'd1_bun_max',

'd1_arterial_ph_max',

'd1_glucose_max',

'd1_wbc_max',

'wbc_apache',

'd1_hemaglobin_max',

'd1_sysbp_noninvasive_max',

'd1_resprate_min',

'd1_creatinine_min',

'd1_calcium_min',

'd1_resprate_max',

'h1_heartrate_min',

'd1_pao2fio2ratio_min',

'd1_pao2fio2ratio_max',

'd1_hco3_max',

'd1_platelets_min',

'd1_lactate_max',

'hematocrit_apache',

'h1_temp_max',

'd1_temp_min',

'd1_arterial_pco2_max',

'bun_apache',

'd1_hematocrit_min',

'creatinine_apache','pre_icu_los_days'].mean()

df_tr = pd.merge(df_tr, grouped, how='left', on=['apache_3j_diagnosis'])

df_ts = pd.merge(df_ts, grouped, how='left', on=['apache_3j_diagnosis'])

df_tr.apache_3j_diagnosis

grouped


train_columns = [x for x in df_tr.columns if x not in ['encounter_id','patient_id','hospital_death','readmission_status']]



categorical_features = []

for m in train_columns:

    if(df_tr[m].dtypes=='object'):

        categorical_features.append(m)

        

df_tr_ts = pd.concat([df_tr[categorical_features],df_ts[categorical_features]])



l = label_var(df_tr_ts, categorical_features)

label_enc(df_tr,l,categorical_features)

label_enc(df_ts,l,categorical_features)



for df in [df_tr, df_ts]:

    for m in categorical_features:

        df[m] = df[m].astype(float)

        

categorical_index = [train_columns.index(x) for x in categorical_features]



target = df_tr['hospital_death']



param = {'task': 'train',

         'boosting': 'gbdt',

         'objective':'binary',

         'metric': 'auc',

         'num_leaves': 15,

         'min_data_in_leaf': 90,

         'learning_rate': 0.01,

         'max_depth': 5,

         'feature_fraction': 0.1,

         'bagging_freq': 1,

         'bagging_fraction': 0.75,

         'use_missing': True,

         'nthread': 4

        }



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=256)

oof = np.zeros(len(df_tr))

r=[]

predictions = np.zeros(len(df_ts))

feature_importance_df = pd.DataFrame()

features = train_columns#[col for col in df_tr.columns if col != 'fc' and col != 'id' and col not in ['hospital_death']]



for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_tr,target.values)):

    strLog = "fold {}".format(fold_)

    print(strLog)

    

    trn_data = lgb.Dataset(df_tr.iloc[trn_idx][train_columns], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(df_tr.iloc[val_idx][train_columns], label=target.iloc[val_idx],reference=trn_data)



    num_round = 7000

    clf = lgb.train(param,trn_data,num_round,valid_sets=val_data,early_stopping_rounds=100,verbose_eval=200,categorical_feature=categorical_index)

    oof[val_idx] = clf.predict(df_tr.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        

    a=roc_auc_score(target.loc[val_idx],clf.predict(df_tr.loc[val_idx,train_columns].values, num_iteration=clf.best_iteration))

    r.append(a)

    

    

    #predictions

    predictions += clf.predict(df_ts[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    

strAUC = roc_auc_score(target, oof)

print(strAUC)

print ("mean: "+str(np.mean(np.array(r))))

print ("std: "+str(np.std(np.array(r))))



df_sub = pd.DataFrame({'encounter_id': df_ts['encounter_id']})

df_sub['hospital_death'] = predictions



df_sub.to_csv("sub1.csv",index=False)









feature_importance_df.head(20)
feature_importance_dfs
feature_importance_df.groupby(['Feature'])[['importance']].mean().sort_values(

        "importance", ascending=False).head(60)
for key, value in feature_importance_dfs.items():

    print('Feature Importance for type ', encoders['type'].classes_[key], ' :')

    print(

    value.groupby(['Feature'])[['importance']].mean().sort_values(

        "importance", ascending=False).head(20)

         )
