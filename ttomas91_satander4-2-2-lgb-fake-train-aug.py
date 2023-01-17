import os

import random

import warnings

import numpy as np

import pandas as pd 

import datetime

from sklearn import preprocessing

from sklearn.metrics import roc_auc_score, roc_curve



import lightgbm as lgb



import seaborn as sns

import matplotlib.patches as patch

import matplotlib.pyplot as plt



warnings.filterwarnings('ignore')



pd.set_option('max_columns', None)



plt.style.use('bmh')

plt.rcParams['figure.figsize'] = (10, 10)

title_config = {'fontsize': 20, 'y': 1.05}

IS_LOCAL = False

if(IS_LOCAL):

    PATH="inputs/"

else:

    PATH="../input/"

os.listdir(PATH)
%%time

train = pd.read_csv(PATH+"train.csv")

test = pd.read_csv(PATH+"test.csv")
train.head()
train.describe()
train.info()
test.head()
test.describe()
test.info()
col=train.columns[2:]

scaler_df = preprocessing.normalize(train.iloc[:,2:], copy=True)

scaled_df = pd.DataFrame(scaler_df,columns=col)

train2=train.iloc[:,:2]

train2[train.columns[2:]]=scaled_df

train=train2

del train2, scaler_df, scaled_df

train.head()
col=test.columns[1:]

scaler_df = preprocessing.normalize(test.iloc[:,1:], copy=True)

scaled_df = pd.DataFrame(scaler_df,columns=col)

test2=test.iloc[:,:1]

test2[test.columns[1:]]=scaled_df

test=test2

del test2, scaler_df, scaled_df

test.head()
col_var = train.columns[2:]

df = pd.DataFrame(col_var, columns=['feature'])

df['n_train_unique'] = train[col_var].nunique(axis=0).values

df['n_test_unique'] = test[col_var].nunique(axis=0).values



for i in df.index:

    col = df.loc[i, 'feature']

    df.loc[i, 'n_overlap'] = int(np.isin(train[col].unique(), test[col]).sum())

    

df['origin_train']=df.n_train_unique-df.n_overlap



df['origin_test']=df.n_test_unique-df.n_overlap

df.T
%%time

for feature in train.columns[2:]:

    train[feature] = np.round(train[feature], 5)

    test[feature] = np.round(test[feature], 5)
col_var = train.columns[2:]

df = pd.DataFrame(col_var, columns=['feature'])

df['n_train_unique'] = train[col_var].nunique(axis=0).values

df['n_test_unique'] = test[col_var].nunique(axis=0).values



for i in df.index:

    col = df.loc[i, 'feature']

    df.loc[i, 'n_overlap'] = int(np.isin(train[col].unique(), test[col]).sum())

    

df['origin_train']=df.n_train_unique-df.n_overlap



df['origin_test']=df.n_test_unique-df.n_overlap

df.T
df = df.sort_values(by='n_train_unique').reset_index(drop=True)

df[['n_train_unique', 'n_test_unique', 'n_overlap']].plot(kind='barh' ,figsize=(22, 100), fontsize=20, width=0.8)

plt.yticks(df.index, df['feature'].values)

plt.xlabel('n_unique', fontsize=20)

plt.ylabel('feature', fontsize=20)

plt.legend(loc='center right', fontsize=20)
df = df.sort_values(by='n_train_unique').reset_index(drop=True)

df[['origin_train', 'origin_test']].plot(kind='barh' ,figsize=(22, 100), fontsize=20, width=0.8)

plt.yticks(df.index, df['feature'].values)

plt.xlabel('n_unique', fontsize=20)

plt.ylabel('feature', fontsize=20)

plt.legend(loc='center right', fontsize=20)
def inform_overlap(train,test,ans,silent=True):

    variable=[c for c in train.columns if c not in ['ID_code','target']]

    vari=[]

    res_train_out=[]

    res_test_out=[]

    res_reclose_train=[]

    res_reclose_test=[]

    for var in variable:

        vari.append(var)

        if silent==False:

            print ('\n Calulate {}'.format(var))

        valu = train[var].isin(test[var].value_counts().index)

        rows = valu[valu==False].index

        

        df_include = train.drop(index=rows)

        df_not_include = train.drop(index=df_include.index)

        

        res_train_out.append(len(df_not_include))

        res_reclose_train.append(len(df_include))

        

        

        valu = test[var].isin(train[var].value_counts().index)

        rows = valu[valu==False].index

        

        df_include = test.drop(index=rows)

        df_not_include = test.drop(index=df_include.index)

        

        res_test_out.append(len(df_not_include))

        res_reclose_test.append(len(df_include))

        

    ans['fetures']=vari

    

    ans['train_out']=res_train_out

    ans['reclose_train']=res_reclose_train

    ans['test_out']=res_test_out

    ans['reclose_test']=res_reclose_test

    

    return ans
%%time

ansver_orig = inform_overlap(train,test,pd.DataFrame())
ansver_orig['dif_train']=ansver_orig.train_out/ansver_orig.reclose_train

ansver_orig['dif_test']=ansver_orig.test_out/ansver_orig.reclose_test

ansver_orig.T
def train_to_test_aug(train,test,silent=True):

    variable=[c for c in train.columns if c not in ['ID_code','target']]

    

    for var in variable:

        if silent==False:

            print ('\n Calulate {}'.format(var))

        valu = train[var].isin(test[var].value_counts().index)

        rows = valu[valu==False].index

        df_include = train.drop(index=rows)

        df_not_include = train.drop(index=df_include.index)



        df_include_True = df_include[df_include.target==True]

        df_include_False = df_include[df_include.target==False]

        df_not_include_True = df_not_include[df_not_include.target==True]

        df_not_include_False = df_not_include[df_not_include.target==False]

        tmp=df_include_True.copy()

        for x in range(len(df_not_include_True)//len(df_include_True)):

            tmp=pd.concat([tmp,df_include_True],ignore_index=True)

        

        if silent==False:

            print ('Target == True:')

            print ("Count row's not include: {} . Count row's exemple: {} .".format(len( df_not_include_True[var]),

                                                                                    len(df_include_True[var])))

     

        df_not_include_True[var]=tmp[var].sample(n=len(df_not_include_True[var])).tolist()

        

        

        tmp=df_include_False.copy()

        for x in range(len(df_not_include_False)//len(df_include_False)+1):

            tmp=pd.concat([tmp,df_include_False],ignore_index=True)

        

        if silent==False:

            print ('Target == False:')

            print ("Count row's not include: {} . Count row's exemple: {} .".format(len( df_not_include_False[var]),

                                                                                    len(df_include_False[var])))

        

        df_not_include_False[var]=tmp[var].sample(n=len(df_not_include_False[var])).tolist()

        

        train=pd.concat([df_include_True,df_include_False,df_not_include_True,df_not_include_False])

        

        train.sort_index(axis=0,inplace = True)

    

    return train
%%time

fake_train = train_to_test_aug(train, test)
col_var = fake_train.columns[2:]

df = pd.DataFrame(col_var, columns=['feature'])

df['n_fake_trane_unique'] = fake_train[col_var].nunique(axis=0).values

df['n_test_unique'] = test[col_var].nunique(axis=0).values



for i in df.index:

    col = df.loc[i, 'feature']

    df.loc[i, 'n_overlap'] = int(np.isin(fake_train[col].unique(), test[col]).sum())

df.T
df = df.sort_values(by='n_test_unique').reset_index(drop=True)

df[['n_fake_trane_unique', 'n_test_unique', 'n_overlap']].plot(kind='barh' ,figsize=(22, 100), fontsize=20, width=0.8)

plt.yticks(df.index, df['feature'].values)

plt.xlabel('n_unique', fontsize=20)

plt.ylabel('feature', fontsize=20)

plt.legend(loc='center right', fontsize=20)
%%time

ansver_fake = inform_overlap(fake_train,test,pd.DataFrame())
ansver_fake['dif_train']=ansver_fake.train_out/ansver_fake.reclose_train

ansver_fake['dif_test']=ansver_fake.test_out/ansver_fake.reclose_test

ansver_fake.T
def augment(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([xs,xn])

    y = np.concatenate([ys,yn])

    return x,y
features= [c for c in fake_train.columns if c not in ['ID_code', 'target']]
%%time



X_t, Y_t = augment(fake_train[features].values, fake_train['target'].values, 8)



tr = pd.DataFrame(X_t)

        

tr = tr.add_prefix('var_')



tr['target'] = Y_t
param_pretrain = {

    'bagging_freq': 1,

    'bagging_fraction': 0.4,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.05,

    'learning_rate': 0.01,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 100,

    'min_sum_hessian_in_leaf': 40.0,

    'num_leaves': 16,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary',

    'verbosity': 1}
%%time

X_t, x_v = tr.iloc[:][features], fake_train.iloc[:][features]

Y_t, y_v = tr.iloc[:]['target'], fake_train.iloc[:]['target']
%%time

sn=datetime.datetime.now()



train_res = np.zeros(len(X_t))

valid_res = np.zeros(len(x_v))



trn_data = lgb.Dataset(X_t, label=Y_t)

val_data = lgb.Dataset(x_v, label=y_v)

val_data2 = lgb.Dataset(train[features], label=train['target'])

        

num_round = 60000

        

evals_result = {}

        

clf=lgb.train(param_pretrain, trn_data, num_round, valid_sets = [trn_data, val_data,val_data2], 

              verbose_eval=1000, early_stopping_rounds = 4000)
%%time

valid_res = clf.predict(x_v, num_iteration=clf.best_iteration)

del1=datetime.datetime.now()-sn

print ("AUG time :  {} . Valid score : {:<8.5f}".format(str(del1), roc_auc_score(y_v,valid_res)))
%%time

tr_result=clf.predict(train[features], num_iteration=clf.best_iteration)

print ("Train score : {:<8.5f}".format(roc_auc_score(train['target'],tr_result)))
%%time

feat_importance_df = pd.DataFrame()

feat_importance_df["feature"] = features

feat_importance_df["importance"] = clf.feature_importance()
%%time



plt.figure(figsize=(14,36))

sns.barplot(x="importance", y="feature", data=feat_importance_df.sort_values(by="importance",ascending=False))

plt.title('Features importance')

plt.tight_layout()

plt.savefig('fet_impotance.png')
%%time

result=clf.predict(test[features], num_iteration=clf.best_iteration)
%%time

sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})

sub_df["target"] = result



sub_df.to_csv("lgb_fake_test_submission{}.csv".format(422), index=False)
%%time

sub_df = pd.DataFrame({"ID_code":train["ID_code"].values})

sub_df["target"] = train["target"].values

sub_df["predict"] = tr_result

sub_df.to_csv("lgb_fake_train_submission{}.csv".format(422), index=False)