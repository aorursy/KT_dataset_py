import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import colors

%matplotlib inline

from matplotlib.colors import Normalize

import seaborn as sns

import operator

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/kc_house_data.csv',index_col='date',parse_dates=True)

train.info()
plt.figure(figsize=(8,8))

plt.hist(train.price,bins=100,color='b')

plt.title('Histogram of House Price')

plt.show()

# 
train['log_price'] = np.log(train['price'])
# monthly change of prices

train['ym'] = (train.index.year *100 + train.index.month).astype(str) 

ym_summary = train.groupby('ym')['price'].agg(['mean','count'])



vmin = np.min(ym_summary['count'])

vmax = np.max(ym_summary['count'])

norm = colors.Normalize(vmin,vmax)



plt.figure(figsize=(15,7))

plt.scatter(x=np.arange(ym_summary.shape[0]), y =ym_summary['mean'],c= ym_summary['count'],

            s= ym_summary['count'],norm=norm ,alpha = 0.8, cmap='jet')



plt.plot(np.arange(ym_summary.shape[0]), ym_summary['mean'] ,'--')

plt.xticks(np.arange(ym_summary.shape[0]),ym_summary.index.values)

plt.yscale('log')

plt.xlabel('Year-Month')

plt.ylabel('Price (log scale)')

clb = plt.colorbar() 

clb.ax.set_title('number of sales')

plt.title('Averge Price by Month')

plt.show()
plt.figure(figsize=(15,10))

vmin = np.min(train.price)

vmax = np.max(train.price)

norm = colors.LogNorm(vmin*2,vmax/3)

plt.scatter(train.long,train.lat, marker='*',c=train.price,norm=norm,cmap='jet') 

plt.xlabel('Longitude')

plt.ylabel('Latituede')

plt.title('House Price by Geography')

clb = plt.colorbar() 

clb.ax.set_title('Price')
corr = train[['sqft_living','sqft_living15','sqft_lot','sqft_lot15','sqft_above','sqft_basement','floors','log_price']].corr()

mask = np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f,ax = plt.subplots(figsize=(8,8))

cmap=sns.diverging_palette(120,10,as_cmap=True) #

sns.heatmap(corr,mask=mask,cmap=cmap,center=0,square=False,linewidths=.5,cbar_kws={"shrink":.5})
def scatter_p(fea,price):

    n_f = len(fea)

    n_row = n_f//3+1

    fig=plt.figure(figsize=(20,15))

    i = 1

    for f in fea:

        x=train[f]

        y=train[price]

        m, b = np.polyfit(x, y, 1)    

        

        ax=fig.add_subplot(n_row,3,i)

        plt.plot(x,y,'.',color='b')

        plt.plot(x, m*x + b, '-',color='r')

        plt.xlabel(f)

        plt.ylabel(price)

        i += 1
scatter_p(fea=['sqft_living', 'sqft_lot','sqft_basement'],price='price')
train['log_sqft_living'] = np.log(train['sqft_living'])

train['log_sqft_lot'] = np.log(train['sqft_lot'])

train['log_sqft_basement'] = np.log1p(train['sqft_basement'])

scatter_p(fea=['log_sqft_living','log_sqft_lot','log_sqft_basement'],price='log_price')
train['basement_ind'] = [1 if x>0 else 0 for x in train.sqft_basement]
scatter_p(fea=['yr_built','yr_renovated','bathrooms'],price='log_price')
train['renovated_ind'] = [1 if x>0 else 0 for x in train.yr_renovated]
x=train.loc[train.renovated_ind==1,'yr_renovated']

y=train.loc[train.renovated_ind==1 ,'log_price']

m, b = np.polyfit(x, y, 1)   

plt.plot(x,y,'.',color='b')

plt.plot(x, m*x + b, '-',color='r')

plt.title('Renovated Houses')

plt.xlabel('yr_renovated')

plt.ylabel('log_price')

plt.show()
corr = train[['bedrooms','condition','grade','view','floors','log_price']].corr()

mask = np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f,ax = plt.subplots(figsize=(8,8))

cmap=sns.diverging_palette(120,10,as_cmap=True) #

sns.heatmap(corr,mask=mask,cmap=cmap,center=0,square=False,linewidths=.5,cbar_kws={"shrink":.5})
fig = plt.figure(figsize=(15,12))

fig.add_subplot(3,2,1)

sns.violinplot(x="bedrooms", y="log_price", data=train)    

fig.add_subplot(3,2,2)

sns.violinplot(x="condition", y="log_price", data=train)  

fig.add_subplot(3,2,3)

sns.violinplot(x="grade", y="log_price", data=train)   

fig.add_subplot(3,2,4)

sns.violinplot(x="view", y="log_price", data=train)  

fig.add_subplot(3,2,5)

sns.violinplot(x="floors", y="log_price", data=train)  
# convert bedrooms, floors, year_month to binary feature

train = pd.get_dummies(train,columns=['bedrooms','floors','ym'],drop_first=True)
wf = train.waterfront.unique()

for i in wf:

    temp_x=train.loc[train.waterfront==i,'log_price']

    ax = sns.kdeplot(temp_x,shade=True)

plt.title('Distribution of log_price by waterfront')
trn = train.drop(['id','price','log_price','sqft_living15','sqft_lot15','sqft_living','sqft_lot','sqft_above',

                  'sqft_basement','zipcode'],axis=1)

resp = train['log_price']
from sklearn.model_selection import train_test_split

import xgboost as xgb

import pickle

from sklearn.metrics import mean_squared_error

import gc

gc.collect()
X_trn, X_tst, y_trn, y_tst = train_test_split(trn,resp,test_size=0.2,random_state=123)
param={

    'objective': 'reg:linear',

    'eta'      :0.02,

    'eval_metric':'rmse',

    'max_depth': 5,

    'min_child_weight':3,

    'subsample' : 0.8,

    'colsample_bytree' : 0.8,

    'silent': 1,

    'seed' : 123

}

trn = xgb.DMatrix(X_trn,label=y_trn)

tst = xgb.DMatrix(X_tst,label=y_tst)

res = xgb.cv(param,trn,nfold=4,num_boost_round=2000,early_stopping_rounds=50,

             stratified=True,show_stdv=True,metrics={'rmse'},maximize=False)

min_index=np.argmin(res['test-rmse-mean'])



model = xgb.train(param,trn,min_index,[(trn,'train'),(tst,'test')])

pred = model.predict(tst)

print('Test RMSE:', np.sqrt(mean_squared_error(y_tst,pred)))
plt.scatter(y_tst,pred,color='b')

plt.xlabel('true log_price')

plt.ylabel('predicted log_price')
r_sq = ((pred-np.mean(y_tst))**2).sum() / ((y_tst - np.mean(y_tst))**2).sum()

print('R square is: ', r_sq)
fig,ax = plt.subplots(1,1,figsize=(10,10))

xgb.plot_importance(model,ax=ax,max_num_features=20)
def partial_dependency(bst, feature):

    X_temp = X_trn.copy()

    grid = np.linspace(np.percentile(X_temp.loc[:, feature], 0.1),np.percentile(X_temp.loc[:, feature], 99.5),50)

    y_pred = np.zeros(len(grid))

    for i, val in enumerate(grid):

            X_temp.loc[:, feature] = val

            data = xgb.DMatrix(X_temp)

            y_pred[i] = np.average(bst.predict(data))

    

    plt.plot(grid,y_pred,'-',color='r')

    plt.plot(X_trn.loc[:,feature], y_trn, 'o',color='grey',alpha=0.01)

    plt.title(('Partial Dependence of '+feature))

    plt.xlabel(feature)

    plt.ylabel('Housing Price (log scale)')

    plt.show()

      
partial_dependency(model,'log_sqft_living')
partial_dependency(model,'lat')
partial_dependency(model,'long')