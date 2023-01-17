from IPython.core.display import display, HTML

from IPython.display import Image

display(HTML("<style>.container { width:80% !important; }</style>"))
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



# Any results you write to the current directory are saved as output./
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

import xgboost as xgb
import matplotlib.pyplot as plt

import time

import seaborn as sns

from pylab import rcParams

%matplotlib inline

#sklearn library
train = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")

test = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")
print ("In train dataset, the number of Records is {}".format(train.shape[0])+", and number of Features is {}".format(train.shape[1]-2)) #not counting ID and target

print ("In test dataset, the number of Records is {}".format(test.shape[0])+", and number of Features is {}".format(test.shape[1]-1))#not counting ID 
train.head()
train.iloc[:,2:].info()
test.head()
test.iloc[:,1:].info()
train.iloc[:,2:]=train.iloc[:,2:].astype(float)
train.iloc[:,2:].info()
train.target.nunique()
plt.figure(figsize=(10, 5))

plt.hist(train.target, bins=50)

plt.title('target Histogram ')

plt.xlabel('Target')

plt.ylabel('Frequency')

plt.show()
plt.figure(figsize=(10, 5))

plt.hist(np.log1p(train.target), bins=50) # equal to "np.log(x+1)"   add 1 to avoid log(0)

plt.title('log target Histogram ')

plt.xlabel('Target')

plt.ylabel('Frequency')

plt.show()
train.target.value_counts().head()
np.log(train.target.median())
print(train.isnull().values.any())

print(test.isnull().values.any())
all_zero_columns=[i for i in train.columns if train[i].nunique()==1]

print ("There are {}".format(len(all_zero_columns))+" all zero columns in train dataset")

print("There is {}".format(len([i for i in test.columns if test[i].nunique()==1]))+" all zero column in test dataset")
def find_duplicate_columns(df):

    duplicate_columns=[]

    for i in range(len(df.columns)):

        this=df.iloc[:,i]

        for j in range(i+1,len(df.columns)):

            compare=df.iloc[:,j]

            if this.equals(compare):

                duplicate_columns.append(train.columns[j])

    return duplicate_columns
#a=find_duplicate_columns(train)

#a=['d60ddde1b', 'acc5b709d', '912836770', 'f8d75792f', 'f333a5f60'] it did take an hour.
train.head()
#use lgbm's parameters I tuned in other kernel 
clf_lgb=lgb.LGBMRegressor(bagging_fraction=0.5, boosting_type='gbdt', class_weight=None,

              colsample_bytree=1.0, feature_fraction=0.5,

              importance_type='split', learning_rate=0.01, max_depth=-1,

              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

              n_estimators=500, n_jobs=-1, num_leaves=130,

              objective='regression', random_state=42, reg_alpha=0.0,

              reg_lambda=0.0, silent=True, subsample=1.0,

              subsample_for_bin=200000, subsample_freq=0)
clf_lgb.fit(np.log1p(train.iloc[:,2:]),np.log1p(train.iloc[:,1]))
fig, ax = plt.subplots(figsize=(14,10))

lgb.plot_importance(clf_lgb, max_num_features=50, height=0.8,color="tomato",ax=ax)

plt.show()
#store the features importance

feat_importances = pd.Series(clf_lgb.booster_.feature_importance(),clf_lgb.booster_.feature_name())

top30=[i for i in feat_importances.nlargest(30).index]
top30.insert(0,'target')

top30.insert(0,'ID')
# build a dataset for rich features

richdf=train[[i for i in top30]]
top30_to_plot =richdf.iloc[:,2:10] .melt(var_name='columns')

g = sns.FacetGrid(top30_to_plot, col='columns')

g = (g.map(sns.distplot, 'value'))
richdf.iloc[:,1:]=np.log1p(richdf.iloc[:,1:])
top30_to_plot =richdf.iloc[:,2:10] .melt(var_name='columns')

g = sns.FacetGrid(top30_to_plot, col='columns')

g = (g.map(sns.distplot, 'value'))
top30_to_plot1 =richdf.iloc[:,10:18] .melt(var_name='columns')

g = sns.FacetGrid(top30_to_plot, col='columns')

g = (g.map(sns.distplot, 'value'))
top30_to_plot['value'] = top30_to_plot['value'].replace(0.0,np.nan)

g = sns.FacetGrid(top30_to_plot.dropna(), col='columns')

g = (g.map(sns.distplot, 'value'))
top30_to_plot1['value'] = top30_to_plot['value'].replace(0.0,np.nan)

g = sns.FacetGrid(top30_to_plot1.dropna(), col='columns')

g = (g.map(sns.distplot, 'value'))
corr=richdf.iloc[:,1:].corr()
#forked from https://www.kaggle.com/samratp/beginner-guide-to-eda-and-modeling

#I have other heatmap but this one is so beautiful!

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(16,16))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title("Correlation HeatMap", fontsize=20)

plt.show()
del richdf,corr,top30_to_plot
# constant columns

all_zero_columns=[i for i in train.columns if train[i].nunique()==1]

train=train[[i for i in train.columns if i not in all_zero_columns]]

test=test[[i for i in test.columns if i not in all_zero_columns]]

# duplicate columns

duplicte_columns=['d60ddde1b', 'acc5b709d', '912836770', 'f8d75792f', 'f333a5f60']

train=train[[i for i in train.columns if i not in duplicte_columns]]

test=test[[i for i in test.columns if i not in duplicte_columns]]

# log transform

X = np.log1p(train.drop(["ID", "target"], axis=1))

y = np.log1p(train["target"].values)

test = np.log1p(test.drop(["ID"], axis=1))
rf=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=20,

                      max_features='auto', max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=0.1, min_samples_split=0.3,

                      min_weight_fraction_leaf=0.0, n_estimators=300,

                      n_jobs=None, oob_score=True, random_state=None, verbose=0,

                      warm_start=False)

Image("../input/imageforscore/rf.png")
Models_score={}

Models_score['Random Forest']=1.69
lgbbest=lgb.LGBMRegressor(bagging_fraction=0.5, boosting_type='gbdt', class_weight=None,

              colsample_bytree=1.0, feature_fraction=0.5,

              importance_type='split', learning_rate=0.01, max_depth=-1,

              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

              n_estimators=500, n_jobs=-1, num_leaves=130,

              objective='regression', random_state=42, reg_alpha=0.0,

              reg_lambda=1, silent=True, subsample=1.0,

              subsample_for_bin=200000, subsample_freq=0)

xgbbest=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.9, gamma=0.3,

             importance_type='gain', learning_rate=0.02, max_delta_step=0,

             max_depth=5, min_child_weight=5, missing=0, n_estimators=500,

             n_job=4, n_jobs=1, nthread=None, objective='reg:squarederror',

             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

             seed=0, silent=None, subsample=0.7, verbosity=1)

Image("../input/imageforscore/lgb.png")
Image("../input/imageforscore/xgb.png")
Models_score['Lightgbm']=1.40192

Models_score['xgboost']=1.43379
Image("../input/imageforscore/stacking_lgb_xgb_rf.png")
Image("../input/imageforscore/softvoting_lgb_xgb_rf.png")
Models_score['softvoting_lgb_xgb_rf']=1.62068

Models_score['stacking_lgb_xgb_rf']=1.45082
Models_score['softvoting_lgb_xgb']=1.39769

Image("../input/imageforscore/softvoting_lgb_xgb.png")
Models_score['stacking_lgb_xgb']=1.5950

Image("../input/imageforscore/stacking_lgb_xgb.png")
modeldf=pd.DataFrame(list(Models_score.items()), columns=['Model', 'RMSE'])

modeldf=modeldf.sort_values('RMSE',ascending = False)
rcParams['figure.figsize'] = 25, 10

rcParams['font.size'] = 15

ax = sns.barplot(x="Model", y="RMSE", data=modeldf)