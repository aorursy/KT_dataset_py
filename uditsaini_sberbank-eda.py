#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

from sklearn import model_selection, preprocessing

import xgboost as xgb

color = sns.color_palette()

warnings.filterwarnings('ignore')

pd.options.display.max_columns=999

from pylab import rcParams

rcParams['figure.figsize'] = 12, 6

%matplotlib inline
df_train=pd.read_csv("../input/train.csv")
#number of samples saize in training data

df_train.shape
df_train.head()
df_train.price_doc.describe()
#skewness and kurtosis

print("Skewness: %f" % df_train['price_doc'].skew())

print("Kurtosis: %f" % df_train['price_doc'].kurt())
print("missing values in each features")

df_train.isnull().sum()
#transection treands by time

import matplotlib.pyplot as plt 

rcParams['figure.figsize'] = 12, 6

df_train.groupby('timestamp').size().plot(kind='line',title="Transection treands by time");

plt.ylabel('Nuber Of Transection')

plt.show()

#Their si peak in the end of 2015 for number of transection
#histogram of target varivale

sns.distplot(df_train['price_doc']);
#Deviate from the normal distribution.

#Have appreciable positive skewness



#skewness and kurtosis

print("Skewness: %f" % df_train['price_doc'].skew())

print("Kurtosis: %f" % df_train['price_doc'].kurt())
#histogram of target varivale

rcParams['figure.figsize'] = 12, 6

sns.distplot(np.log1p(df_train['price_doc']));
print("missing values in each features")

missingvalues=df_train.isnull().sum()

missingvalues
##lets look at what feature has highest missing values

from pylab import rcParams

rcParams['figure.figsize'] = 10, 15

pd.DataFrame(missingvalues[missingvalues>0]).plot(kind='barh');

plt.ylabel('Feature Name')

plt.xlabel('missing values count');

plt.show()
###lest look at the numerical variables and its correlation with target 
df_train_num=df_train.select_dtypes(exclude=[np.object])

nummissingvalues=df_train_num.isnull().sum()

col_missingval_lesstehn500=list(nummissingvalues[nummissingvalues<500].index)

col_missingval_lesstehn500.remove('id')

#col_missingval_lesstehn500.remove('timestamp')



##highly correleted vartiables 

mcorr = df_train_num[col_missingval_lesstehn500].corr().abs()

mcorr = mcorr.unstack()

mcorr = mcorr.order(kind="quicksort",ascending=False)

pd.DataFrame(mcorr).head(10)

#high correleted variables with price_doc/target column with sorted order

targetcorr=pd.DataFrame(mcorr['price_doc'][1:])

targetcorr.columns=['correlation']

targetcorr.head(10)
#Relationship with numerical variables

#scatter plot full_sq/price_doc

from pylab import rcParams

rcParams['figure.figsize'] = 12, 6

var = 'full_sq'

df_train_num.plot.scatter(x=var, y='price_doc');
#Relationship with numerical variables

#scatter plot sport_count_5000	/price_doc

var = 'sport_count_5000'

df_train_num.plot.scatter(x=var, y='price_doc');
#Relationship with numerical variables

#scatter plot sport_count_3000	/price_doc

var = 'sport_count_3000'

df_train_num.plot.scatter(x=var, y='price_doc');
#Relationship with numerical variables

#scatter plot trc_count_5000	/price_doc

var = 'trc_count_5000'

df_train_num.plot.scatter(x=var, y='price_doc');
#Relationship with numerical variables

#scatter plot zd_vokzaly_avto_km	/price_doc

var = 'zd_vokzaly_avto_km'

df_train_num.plot.scatter(x=var, y='price_doc');
#Relationship with numerical variables

#scatter plot zd_vokzaly_avto_km	/price_doc

var = 'sadovoe_km'

df_train_num.plot.scatter(x=var, y='price_doc');
#Relationship with numerical variables

#scatter plot zd_vokzaly_avto_km	/price_doc

var = 'kremlin_km'

df_train_num.plot.scatter(x=var, y='price_doc');
#Relationship with numerical variables

#scatter plot zd_vokzaly_avto_km	/price_doc

var = 'bulvar_ring_km'

df_train_num.plot.scatter(x=var, y='price_doc');
#Relationship with numerical variables

#scatter plot zd_vokzaly_avto_km	/price_doc

var = 'sport_count_2000'

df_train_num.plot.scatter(x=var, y='price_doc');
#Relationship with numerical variables

#scatter plot zd_vokzaly_avto_km	/price_doc

var = 'ttk_km'

df_train_num.plot.scatter(x=var, y='price_doc');
#correlation of top 40 variables with price_doc

corrmat=df_train_num[targetcorr.head(30).index].corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
col=list(df_train.select_dtypes(include=[np.object]).columns)

rcParams['figure.figsize'] = 12, 6

#Relationship with categorical features

#box plot sub_area/price_doc

var = 'sub_area'

sns.boxplot(x=var, y="price_doc", data=df_train);
#Relationship with categorical features

#box plot product_type/price_doc

var = 'product_type'

sns.violinplot(x=var, y="price_doc", data=df_train);
#Relationship with categorical features

#box plot ecology/price_doc

var = 'ecology'

sns.violinplot(x=var, y="price_doc", data=df_train);
#Relationship with categorical features

#box plot railroad_terminal_raion/price_doc

var = 'railroad_terminal_raion'

sns.violinplot(x=var, y="price_doc", data=df_train);
#Relationship with categorical features

#box plot product_type/price_doc

var = 'culture_objects_top_25'

sns.violinplot(x=var, y="price_doc", data=df_train);
#source https://www.kaggle.com/sudalairajkumar/sberbank-russian-housing-market/simple-exploration-notebook-sberbank

for f in df_train.columns:

    if df_train[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df_train[f].values)) 

        df_train[f] = lbl.transform(list(df_train[f].values))

        

train_y = df_train.price_doc.values

train_X = df_train.drop(["id", "timestamp", "price_doc"], axis=1)



xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)



# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()