

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

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import datetime as dt

plt.style.use('ggplot')
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv',parse_dates=['datetime'])

test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv',parse_dates=['datetime'])
train.head()
test.head()
import missingno as msno

msno.matrix(train)
msno.matrix(test)
train.info()
train['year'] = train['datetime'].dt.year

train['month'] = train['datetime'].dt.month

train['day'] = train['datetime'].dt.day

train['hour'] = train['datetime'].dt.hour

train['weekday'] = train['datetime'].dt.weekday

train.head()
fig,axes = plt.subplots(2,2,figsize=(12,8))

sns.barplot(x=train['year'],y=train['count'],ax=axes[0,0],ci=False)

sns.barplot(x=train['month'],y=train['count'],ax=axes[0,1],ci=False)

#sns.barplot(x=train['day'],y=train['count'],ax=axes[1,0],ci=False)

sns.barplot(x=train['hour'],y=train['count'],ax=axes[1,0],ci=False)

sns.barplot(x=train['weekday'],y=train['count'],ax=axes[1,1],ci=False)

axes[0,0].set(title='Bike sharing count by year')

axes[0,1].set(title='Bike sharing count by month')

axes[1,0].set(title='Bike sharing count by hour')

axes[1,1].set(title='Bike sharing count by weekday')

plt.tight_layout()
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(12,8))



sns.boxplot(data=train,y="count",orient="v",ax=axes[0][0])

sns.boxplot(data=train,y="count",x="season",orient="v",ax=axes[0][1])

sns.boxplot(data=train,y="count",x="hour",orient="v",ax=axes[1][0])

sns.boxplot(data=train,y="count",x="workingday",orient="v",ax=axes[1][1])



axes[0][0].set(ylabel='Count',title="Box Plot On Count")

axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")

axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")

axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
plt.figure(figsize=(12,8))

corr_matrix = train[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()

mask = np.zeros_like(corr_matrix)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_matrix,mask=mask,vmax=.8,square=True,annot=True)
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(12,8))

sns.regplot(x=train['temp'],y=train['count'],ax=ax1)

sns.regplot(x=train['humidity'],y=train['count'],ax=ax2)

sns.regplot(x=train['windspeed'],y=train['count'],ax=ax3)
fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5,figsize=(15,25))

sns.pointplot(data=train,x='hour',y='count',ax=ax1)

sns.pointplot(data=train,x='hour',y='count',hue='workingday',ax=ax2)

sns.pointplot(data=train,x='hour',y='count',hue='weekday',ax=ax3)

sns.pointplot(data=train,x='hour',y='count',hue='season',ax=ax4)

sns.pointplot(data=train,x='hour',y='count',hue='weather',ax=ax5)
from scipy.stats import norm, skew

from scipy import stats

plt.style.use('seaborn')

sns.distplot(train['count'] , fit=norm)

mu, sigma = norm.fit(train['count'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')

plt.ylabel('Frequency')

plt.title('Count distribution')



#Get also the QQ-plot

fig = plt.figure()

stats.probplot(train['count'],plot=plt)

plt.show()
sns.distplot(np.log1p(train['count']),fit=norm)

mu,sigma = norm.fit(np.log1p(train['count']))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')

plt.ylabel('Frequency')

plt.title('count distribution')

fig=plt.figure()

stats.probplot(np.log1p(train['count']),fit=True,plot=plt)

plt.show()
data= train.append(test)

windColumns = ["season","weather","humidity","month","temp","year","atemp"]

data['year'] = data['datetime'].dt.year

data['month'] = data['datetime'].dt.month

data['day'] = data['datetime'].dt.day

data['hour'] = data['datetime'].dt.hour

data['weekday'] = data['datetime'].dt.weekday

data['dayofweek'] = data['datetime'].dt.dayofweek
from xgboost import XGBRegressor

X = data[data['windspeed']!=0]

y = data[data['windspeed']==0]

wind_train_x = X[windColumns]

wind_train_y = X['windspeed']

wind_test_x = y[windColumns]

wind_test_y_idx = y['windspeed'].index

xgb=XGBRegressor()

xgb.fit(wind_train_x,wind_train_y)

pred = xgb.predict(wind_test_x)
y['windspeed'] = pred

data = X.append(y).sort_values('datetime')
data[data['windspeed'] == 0]
plt.figure(figsize=(12,8))

sns.countplot(np.round(data['windspeed']))

plt.xticks(rotation=60)

plt.title('windspeed countplot(int)')
category_features = ["season","holiday","workingday","weather","weekday","year"]

for i in category_features:

    data[i] = data[i].astype('category')

final_train = data[data['count'].notnull()]

final_test = data[data['count'].isnull()]

train_x = final_train.drop('count',axis=1)

train_y = final_train['count']

test_x = final_test.drop('count',axis=1)

datetime = test_x.datetime
drop_feat = ['datetime','day','casual','registered']

train_x.drop(drop_feat,axis=1,inplace=True)

test_x.drop(drop_feat,axis=1,inplace=True)
dummy_train_x = pd.get_dummies(train_x)

dummy_test_x = pd.get_dummies(test_x)
from sklearn.metrics import mean_squared_log_error

def rmsle(pred_y,test_y):    

    return np.sqrt(mean_squared_log_error(test_y,pred_y))
from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import RidgeCV,LassoCV,ElasticNetCV

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.model_selection import GridSearchCV

from lightgbm.sklearn import LGBMRegressor
alpha_las=[0.0005,0.0001,0.00005,0.00001]

e_ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

lasso = make_pipeline(RobustScaler(),LassoCV(alphas=alpha_las,random_state=42,max_iter=1e7))

ridge = make_pipeline(RobustScaler(),RidgeCV(alphas = alpha_las))

elastic = make_pipeline(RobustScaler(),ElasticNetCV(max_iter=1e7,alphas=alpha_las,l1_ratio = e_ratio))

rf = RandomForestRegressor(bootstrap=True,max_depth=70,max_features='auto',min_samples_leaf=4,min_samples_split=10,n_estimators=2200)

gra = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

lgbm = LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

ada = AdaBoostRegressor(n_estimators=2200,random_state=42,learning_rate=0.05)
model = [lasso,ridge,elastic,rf,gra,xgb,lgbm,ada]

model_name = ['Lasso','Ridge','ElasticNet','RandomForest','GradientBoost','XGBoost','LGBM','Ada']

tmp = pd.DataFrame(columns=['model','rmsle'])

idx=0

for i,j in zip(model,model_name):

    train_y_log = np.log1p(train_y)

    i.fit(dummy_train_x,train_y_log)

    pred = i.predict(dummy_train_x)

    tmp.loc[idx,'model'] = j

    tmp.loc[idx,'rmsle'] = rmsle(train_y_log,pred)

    idx+=1

tmp = tmp.sort_values(by= 'rmsle')

tmp
from mlxtend.regressor import StackingCVRegressor

stack = StackingCVRegressor(regressors=(gra,xgb,lgbm,ada),

                           meta_regressor=rf,use_features_in_secondary=True)
train_y_log = np.log1p(train_y)

stack.fit(np.array(dummy_train_x),np.array(train_y_log))

pred = stack.predict(np.array(dummy_train_x))

tmp.loc[idx,'model'] = 'Stack'

tmp.loc[idx,'rmsle'] = rmsle(train_y_log,pred)

tmp = tmp.sort_values('rmsle')
tmp['rmsle2'] = tmp['rmsle'].map('{:.4f}'.format)
plt.figure(figsize=(12,8))

ax = fig.add_subplot()

plt.plot(tmp['model'],tmp['rmsle2'])

for i,j in zip(tmp['model'],tmp['rmsle2']):

#     ax.annotate(str(j),xy=(i,j))

    plt.text(i, j, str(j),fontsize=15)

plt.xticks(rotation=60)

plt.title('RMSLE score by model(train data)')

plt.show()
def blend_model(X):

    return (0.3*rf.predict(X))+(0.25*stack.predict(np.array(X)))+(0.2*gra.predict(X))+(0.1*xgb.predict(X))+(0.1*lgbm.predict(X))+(0.05*ada.predict(X))
pred = blend_model(dummy_train_x)

print(rmsle(train_y_log,pred))
# final = pd.DataFrame(columns=['datetime','count'])

# final['datetime'] = datetime

# final['count'] = np.exp(blend_model(dummy_test_x))

# final.to_csv('blend bike submission.csv',index=False)

#0.41146
final = pd.DataFrame(columns=['datetime','count'])

final['datetime'] = datetime

final['count'] = np.exp(stack.predict(np.array(dummy_test_x)))

final.to_csv('rf bike submission.csv',index=False)

# 0.404