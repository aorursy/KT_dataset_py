

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,make_scorer

from sklearn.ensemble import RandomForestRegressor

from hyperopt import tpe,hp,Trials

from hyperopt.fmin import fmin

import xgboost as xgb

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))



adm_pred1=pd.read_csv('../input/Admission_Predict.csv',index_col=0)

adm_pred1_1=pd.read_csv('../input/Admission_Predict_Ver1.1.csv',index_col=0)
data=pd.concat([adm_pred1,adm_pred1_1])

data.sample(5)
data.shape
data.info()
data.describe()
sns.distplot(data['GRE Score'],kde=True)
sns.distplot(data['TOEFL Score'],kde=True)
sns.distplot(data['CGPA'],kde=True)
data.rename(columns={'LOR ':'LOR','Chance of Admit ':'CoA'},inplace=True)
plt.subplots(2,2,figsize=(10,8))

plt.subplot(2,2,1)

sns.countplot(data['SOP'])

plt.subplot(2,2,2)

sns.countplot(data['LOR'])

plt.subplot(2,2,3)

sns.countplot(data['University Rating'])

plt.subplot(2,2,4)

sns.countplot(data['Research'])



plt.tight_layout()
data['CoA'].plot.hist()


sns.lmplot(x='GRE Score',y='CoA',data=data)

sns.jointplot(x='TOEFL Score',y='CoA',data=data)
data.plot.scatter(x='CGPA',y='CoA')
plt.subplots(2,2,figsize=(10,10))

plt.subplot(2,2,1)

sns.violinplot(x='SOP',y='CoA',data=data)

plt.subplot(2,2,2)

sns.boxplot(x='LOR',y='CoA',data=data)

plt.subplot(2,2,3)

sns.stripplot(x='University Rating',y='CoA',data=data)

plt.subplot(2,2,4)

sns.boxenplot(x='Research',y='CoA',data=data)
data.isnull().sum()




X=data.drop('CoA',axis=1)

Y=data['CoA']

train_X,val_X,train_y,val_y=train_test_split(X,Y,test_size=0.2,random_state=1)



lr=LinearRegression()

lr.fit(train_X,train_y)

pred=lr.predict(val_X)

score=mean_squared_error(val_y,pred)

print(score)




rfr=RandomForestRegressor(random_state=1)

rfr.fit(train_X,train_y)

pred_rfr=rfr.predict(val_X)

score_rfr=mean_squared_error(pred_rfr,val_y)

print(score_rfr)




Xgb=xgb.XGBRegressor(random_state=1)

Xgb.fit(train_X,train_y)

pred_xgb=Xgb.predict(val_X)

score_xgb=mean_squared_error(val_y,pred_xgb)

print(score_xgb)




seed=2

def objective(params):

    est=int(params['n_estimators'])

    md=int(params['max_depth'])

    msl=int(params['min_samples_leaf'])

    mss=int(params['min_samples_split'])

    model=RandomForestRegressor(n_estimators=est,max_depth=md,min_samples_leaf=msl,min_samples_split=mss)

    model.fit(train_X,train_y)

    pred=model.predict(val_X)

    score=mean_squared_error(val_y,pred)

    return score



def optimize(trial):

    params={'n_estimators':hp.uniform('n_estimators',100,500),

           'max_depth':hp.uniform('max_depth',5,20),

           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),

           'min_samples_split':hp.uniform('min_samples_split',2,6)}

    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.RandomState(seed))

    return best



trial=Trials()

best=optimize(trial)





        

    
print(best)
for t in trial.trials[:2]:

    print (t)
TID=[t['tid'] for t in trial.trials]

Loss=[t['result']['loss'] for t in trial.trials]

maxd=[t['misc']['vals']['max_depth'][0] for t in trial.trials]

nest=[t['misc']['vals']['n_estimators'][0] for t in trial.trials]

min_ss=[t['misc']['vals']['min_samples_split'][0] for t in trial.trials]

min_sl=[t['misc']['vals']['min_samples_leaf'][0] for t in trial.trials]



hyperopt_rfr=pd.DataFrame({'tid':TID,'loss':Loss,

                          'max_depth':maxd,'n_estimators':nest,

                          'min_samples_split':min_ss, 'min_samples_leaf':min_sl})



plt.subplots(3,2,figsize=(10,10))

plt.subplot(3,2,1)

sns.scatterplot(x='tid',y='max_depth',data=hyperopt_rfr)

plt.subplot(3,2,2)

sns.scatterplot(x='tid',y='loss',data=hyperopt_rfr)

plt.subplot(3,2,3)

sns.scatterplot(x='tid',y='n_estimators',data=hyperopt_rfr)

plt.subplot(3,2,4)

sns.scatterplot(x='tid',y='min_samples_leaf',data=hyperopt_rfr)

plt.subplot(3,2,5)

sns.scatterplot(x='tid',y='min_samples_split',data=hyperopt_rfr)



plt.tight_layout()


seed=5

def objective2(params):

    est=int(params['n_estimators'])

    md=int(params['max_depth'])

    learning=params['learning_rate']

    

    

    model=xgb.XGBRegressor(n_estimators=est,max_depth=md,learning_rate=learning)

    model.fit(train_X,train_y)

    pred=model.predict(val_X)

    score=mean_squared_error(val_y,pred)

    return score



def optimize2(trial):

    params={'n_estimators':hp.uniform('n_estimators',100,500),

           'max_depth':hp.uniform('max_depth',5,20),

           'learning_rate':hp.uniform('learning_rate',0.01,0.1)}

    best2=fmin(fn=objective2,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.RandomState(seed))

    return best2



trial2=Trials()

best2=optimize2(trial2)
print(best2)
TID2=[t['tid'] for t in trial2.trials]

Loss2=[t['result']['loss'] for t in trial2.trials]

maxd2=[t['misc']['vals']['max_depth'][0] for t in trial2.trials]

nest2=[t['misc']['vals']['n_estimators'][0] for t in trial2.trials]

lrt=[t['misc']['vals']['learning_rate'][0] for t in trial2.trials]





hyperopt_xgb=pd.DataFrame({'tid':TID2,'loss':Loss2,

                          'max_depth':maxd2,'n_estimators':nest2,

                          'learning_rate':lrt})
plt.subplots(2,2,figsize=(10,10))

plt.subplot(2,2,1)

sns.scatterplot(x='tid',y='max_depth',data=hyperopt_xgb)

plt.subplot(2,2,2)

sns.scatterplot(x='tid',y='loss',data=hyperopt_xgb)

plt.subplot(2,2,3)

sns.scatterplot(x='tid',y='n_estimators',data=hyperopt_xgb)

plt.subplot(2,2,4)

sns.scatterplot(x='tid',y='learning_rate',data=hyperopt_xgb)





plt.tight_layout()


rfr_opt=RandomForestRegressor(n_estimators=151,max_depth=17,min_samples_split=2,min_samples_leaf=1)

rfr_opt.fit(train_X,train_y)

pred_rfr_opt=rfr_opt.predict(val_X)

score_rfr_opt=mean_squared_error(val_y,pred_rfr_opt)

print(score_rfr_opt)



xgb_opt=xgb.XGBRegressor(n_estimators=427,max_depth=9,learning_rate=0.06446)

xgb_opt.fit(train_X,train_y)

pred_xgb_opt=xgb_opt.predict(val_X)

score_xgb_opt=mean_squared_error(val_y,pred_xgb_opt)

print(score_xgb_opt)