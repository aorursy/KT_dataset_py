from IPython.display import HTML,YouTubeVideo

display(YouTubeVideo('B2CHNrNmM80', width=600, height=300))
# 1.1 Data manipulation libraries

import pandas as pd

import numpy as np



# Dimensionality reduction

from sklearn.decomposition import KernelPCA



# Data transformation classes

from sklearn.preprocessing import OneHotEncoder as ohe

from sklearn.preprocessing import LabelEncoder as le

from sklearn.preprocessing import StandardScaler as ss

 

#Data splitting

from sklearn.model_selection import TimeSeriesSplit



#Model pipelining

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



#Model



from skopt import BayesSearchCV

from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import TimeSeriesSplit as tss



#Plotting

import matplotlib.pyplot as plt

import seaborn as sns



# Other small utilities

from sklearn.metrics import make_scorer

from pandas.tseries.offsets import MonthEnd



import gc

import datetime



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import warnings

warnings.filterwarnings('ignore')



import os
os.chdir('/kaggle/input/rossmann-store-sales/')

train = pd.read_csv("train.csv",parse_dates=[2])

test = pd.read_csv("test.csv",parse_dates=[3])

store = pd.read_csv("store.csv")
train.info()

display(HTML('<h3>Features in train having null values:</h3>'))

train.columns.values[train.isnull().any()]

display(HTML('<h3>Features wise Minimum-Maximum in training dataset:</h3>'))

pd.DataFrame([train.min(),train.max()])
train['Date']=train.Date.astype('datetime64[D]')

train['Store']=train.Store.astype('category')

train['DayOfWeek'] = train.DayOfWeek.astype('category')
train.head()
store.info()

display(HTML('<h3>Features wise Minimum-Maximum and NaN in Store dataset:</h3>'))

pd.DataFrame([store.min(),store.max(),store.isnull().sum(),store.nunique()],index=['Min','Max','Nulls','Unique'])
store.CompetitionDistance.fillna(store.CompetitionDistance.mean(),inplace=True)

store[store.Promo2SinceWeek.isnull()].describe(include='all',percentiles=[])
display(HTML('<h4>From table we get that the rest null values occur due to no promo2 are run on some stores, thus we can put a constant value 0 there'))

store.fillna(0,inplace=True)
store.head()
X=train.merge(store,on='Store',copy=False)
_=plt.figure(figsize=(20,5))

_=train.set_index(keys='Date',drop=False).resample('M')['Sales'].sum().plot(fontsize=20)

_=plt.xlabel('Date', fontsize=20)

_=plt.ylabel('Sales', fontsize=20)

_=plt.suptitle('Rossmann Stores Sales over time', fontsize=30)
_=train.set_index(keys='Date',drop=False).groupby('Store').resample('M')['Sales'].sum().reset_index(level=[0,1])

f,ax=plt.subplots(10,1,sharex=True)

ax=ax.flatten()

for i in range(10):

    __=_[_.Store==(i+1)].plot(x='Date',y='Sales',legend=False,title='Store'+str(i),ax=ax[i],figsize=(20,20))

del _

del __

gc.collect()
%%time

display(HTML('<h4>Now we see week-of-day wise sales for all stores</h4>'))

_=train.groupby('DayOfWeek').agg({'Sales':np.mean}).plot(kind='bar',color='bgyr',legend=[])

__=_.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
display(HTML('The graph above poses 2 big questions:<br>1- Sunday having least sales!Why?<br>2- Whether it is just amount or customer footfall decrease?'))



_=train.plot.scatter('Customers','Sales',s=20,c='DayOfWeek',cmap='rainbow',figsize=(10,5),alpha=0.6)
train.StateHoliday.replace({0:'0'},inplace=True)

f,(ax1,ax2)=plt.subplots(1,2,figsize=(20,5))

_=sns.heatmap(train.groupby(['DayOfWeek','StateHoliday']).agg({'Customers':np.mean}).unstack().fillna(0),cmap='GnBu',ax=ax1,annot=True)

_=sns.heatmap(train.groupby(['SchoolHoliday','DayOfWeek']).agg({'Customers':np.mean}).unstack().fillna(0),cmap='GnBu',ax=ax2,annot=True)

_=ax1.set_title('State Holiday Vs Day of Week')

_=ax2.set_title('School Holiday Vs Day of Week')
_=train.merge(store, on='Store').groupby('Assortment').agg({'Customers':np.mean,'Sales':np.mean})

_["SalesPerCustomer"]=_.Sales/_.Customers

_
f,ax=plt.subplots(1,2,figsize=(20,5))

_=sns.distplot(train.Customers,ax=ax[0])

_=sns.distplot(train.Sales,ax=ax[1])

ax[0].set_title('Customers Distribution')

ax[1].set_title('Sales Distribution')
train['TicketSize']= train['Sales'] / train['Customers']

med_sales= train.groupby('Store')[['Sales', 'Customers', 'TicketSize']].median()

med_sales.rename(columns=lambda x: x+'_median', inplace=True)

train.drop(columns=['TicketSize'],inplace=True)

#med_sales.sample(5)



def build_features(train):

    X= train.merge(med_sales,on='Store')

    X = X.merge(store,on='Store')

    X['Year'] = X.Date.dt.year

    X['Month'] = X.Date.dt.month

    X['Day'] = X.Date.dt.day

    X['Q_Month'] = (train.Date.dt.month-1)%3+1

    X['CompOpSinceMonth']=(X.Year-X.CompetitionOpenSinceYear)*12+(X.Month-X.CompetitionOpenSinceMonth)

    X['LeftDaysInMonth'] = ((X.Date+MonthEnd(0))-X.Date).dt.days

    

    #store.PromoInterval.astype('category').cat.categories

    cat1 = pd.CategoricalDtype(categories=[0, 'Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec'])

    X['PromoInterval'] = X.PromoInterval.astype(cat1).cat.codes

    

    ##Change types

    cat_cols = ['Store', 'DayOfWeek', 'Open','Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment','Promo2','Q_Month','PromoInterval']

    int_cols = ['CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 'Promo2SinceWeek','Promo2SinceYear', 'Year', 'Day', 'CompOpSinceMonth', 'LeftDaysInMonth']

    for col in cat_cols:

        X[col] = X[col].astype('category')

    for col in int_cols:

        X[col] = X[col].astype('int')

    return X
X= build_features(train)

y=X.pop('Sales')

y = np.log1p(y)
def rmspe_log1p(y,yhat):

    y=np.expm1(y)

    yhat=np.expm1(yhat)

    weight=pd.Series([1/a if a!=0 else 0 for a in y])

    return np.sqrt(np.mean((weight*(yhat-y))**2))



rmspe_scorer = make_scorer(rmspe_log1p, greater_is_better = False)
mCat_cols = ['Store','DayOfWeek','StateHoliday','StoreType', 'Assortment','Q_Month','PromoInterval']

bin_cat_cols = ['Open', 'Promo','SchoolHoliday','Promo2']

num_cols = X.select_dtypes('number').columns.to_list()
ctt = ColumnTransformer(

                        [

                            ('mcat',ohe(),mCat_cols),

                            ('num',ss(),num_cols)

                        ])

X_t=ctt.fit_transform(X)
xgboost_tree = XGBRegressor(

    n_jobs = -1,

    n_estimators = 1000,

    eta = 0.1,

    max_depth = 2,

    min_child_weight = 2,

    subsample = 0.8,

    colsample_bytree = 0.8,

    tree_method = 'exact',

    reg_alpha = 0.05,

    random_state = 1023

)

xgboost_tree.fit(X_t, y,

                 eval_metric = rmspe_log1p

                )
rmspe_log1p(y,xgboost_tree.predict(X_t))
import xgboost as xgb

def rmspe_xg(yhat, y):

    y = np.expm1(y.get_label())

    yhat = np.expm1(yhat)

    

    weight=pd.Series([1/a if a!=0 else 0 for a in y])

    return "rmspe", -np.sqrt(np.mean((weight*(yhat-y))**2))



dtrain = xgb.DMatrix(X_t, y)



params = {

    'n_estimators': (200, 2000),

    'max_depth' :(1,8),

    'eta':(0.01, 0.6, 'log-uniform'),

    'colsample_bytree':(0.1,0.9,'uniform'),

    'gamma':(1,10),

    'alpha':(0,10),

    'lambda':(1,10),

    'subsample':(0.1,1.0),

    'min_child_weight':(0,5)

}



bayes_cv = BayesSearchCV(

                        estimator = XGBRegressor(objective= 'reg:linear',

                                                 booster='gbtree',

                                                 verbosity=2,

                                                 tree_method='hist',

                                                 feval=rmspe_xg

                                                ),

                        search_spaces = params, 

                        cv=tss(3),

                        n_jobs=-1,

                        n_iter = 100,

                        verbose=0

                        )
%%time

bayes_cv.fit(X_t,y.values)
bayes_cv.best_score_
rmspe_log1p(bayes_cv.predict(X_t),y)
bayes_cv.best_params_
rand_stores = np.random.randint(0,1115,5)

_ = pd.DataFrame(train[['Date','Store','Sales']])

_['Prediction'] = bayes_cv.predict(X_t)

_['Prediction'] = np.expm1(_.Prediction)

_=_.set_index(keys='Date',drop=False).groupby('Store').resample('M')['Sales','Prediction'].sum().reset_index(level=[0,1])

f,ax=plt.subplots(5,1,sharex=True)

ax=ax.flatten()

for i in range(5):

    __=_[_.Store==rand_stores[i]].plot(x='Date',y='Sales',title='Store'+str(rand_stores[i]),ax=ax[i],figsize=(20,20))

    __=_[_.Store==rand_stores[i]].plot(x='Date',y='Prediction',title='Store'+str(rand_stores[i]),ax=ax[i])

    

del _

del __

gc.collect()