# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import explained_variance_score, roc_auc_score, precision_recall_curve, roc_curve, average_precision_score,accuracy_score
from sklearn import model_selection, preprocessing, metrics
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score, cross_validate

##from keras starter
#from keras.layers import Dense,Dropout
#from keras.models import Sequential
#from keras.optimizers import SGD,RMSprop
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
#from keras.layers.normalization import BatchNormalization
#from keras import backend as K


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import gc
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def load_df(csv_path='../input/train.csv', low_memory=False, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
%%time
##One time activity if you export flattened files to CSV to load later

train = load_df()
test = load_df('../input/test.csv')
train.head()
test.head()
gc.collect()
%%time
##One time activity if you export flattened files to CSV to load later on your local environment

train.to_csv("train-flattened.csv", index=False)
test.to_csv("test-flattened.csv", index=False)
%%time
train_flat = pd.read_csv("train-flattened.csv", low_memory=False, nrows=903653)
test_flat = pd.read_csv("test-flattened.csv", low_memory=False, nrows=804684)
del train
del test
gc.collect()
train_flat.head()
train_flat.describe()
print(train_flat.info(), test_flat.info())

#TEST - dtypes: bool(1), float64(4), int64(6), object(42)
#TRAIN - dtypes: bool(1), float64(5), int64(6), object(43) -- extra column is [trafficSource.campaignCode]
##On the read csv of the flattened file there are more numerics!
%%time
objcol = test_flat.columns

##For some reason [trafficSource.campaignCode] is not present in the TEST dataset.
##already checked the vlaues using --------- train_flat['trafficSource.campaignCode'].value_counts(), and there is only 1 value with a count of 1 rest is nan!

for col in objcol:
    train_u = train_flat[col].unique()
    train_ucnt = train_flat[col].nunique()
    test_u = test_flat[col].unique()
    test_ucnt = test_flat[col].nunique() 
    train_na = train_flat[col].isna().sum()
    test_na = test_flat[col].isna().sum()

    if train_flat[col].nunique() <= 20:
        print(col, ' - ', train_flat[col].dtypes , ' - TRAIN - ', round(train_na * 100 / 903653,2), '% is NAN', '--- TOTAL NAN', train_na,  '    ====== UNIQUE VALUES TRAIN-   ', train_ucnt, '     ======', train_u)
        print('                              TEST - ', round(test_na * 100 / 804684,2), '% is NAN', '--- TOTAL NAN', test_na,  '    ====== UNIQUE VALUES TEST-   ', test_ucnt, '     ======', test_u, '\n')
    else:
        print(col, ' - ', train_flat[col].dtypes , ' - TRAIN - ', round(train_na * 100 / 903653,2), '% is NAN', '--- TOTAL NAN', '    ====== UNIQUE VALUES TRAIN-   ', train_ucnt, '     ====== TOO MANY VALUES TO PRINT!!')
        print('                              TEST - ', round(test_na * 100 / 804684,2), '% is NAN', '--- TOTAL NAN',  '    ====== UNIQUE VALUES TEST-   ', test_ucnt, '     ====== TOO MANY VALUES TO PRINT!!\n')
##Fillna with 0
train_flat['totals.transactionRevenue'].fillna(0,inplace=True)

def fillNan(cols):
    for col in cols:
        train_flat[col].fillna(0,inplace=True)
        test_flat[col].fillna(0,inplace=True)
    
cols = ['trafficSource.adwordsClickInfo.page', 'trafficSource.isTrueDirect', 'totals.newVisits', 'totals.bounces', 'trafficSource.adwordsClickInfo.isVideoAd']
fillNan(cols);  
##Create new column TransactionRevenueLog and transform Date
#had previously set it to np.og but changed to np.log1p to handle ,0, values in revenue based on the dicussion here https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47124
train_flat['totals.transactionRevenueLog'] =  np.log1p(train_flat['totals.transactionRevenue'])

##Create a new NAN column to help with teh reporting aspect as '0' shows up on all the plots and makes it relly hard to see the distribution of the actual transactions with values in there.
train_flat['totals.transactionRevenueLogNAN'] =  np.log1p(train_flat['totals.transactionRevenue'])
train_flat['totals.transactionRevenueLogNAN'].replace(0,np.nan,inplace=True)
train_flat['totals.transactionRevenueLogNAN'].head()
##transform the date columns
train_flat['datestr'] = pd.to_datetime(train_flat['date'].astype('str'), format='%Y%m%d')
test_flat['datestr'] = pd.to_datetime(test_flat['date'].astype('str'), format='%Y%m%d')

##technically the .dt.day, dt.month should work in Kaggle but it does not seem to!

train_flat['year'], train_flat['month'],train_flat['day'], train_flat['week']  = train_flat['datestr'].apply(lambda x: x.year).astype('int64'), train_flat['datestr'].apply(lambda x: x.month).astype('int64'), train_flat['datestr'].apply(lambda x: x.day).astype('int64'), train_flat['datestr'].apply(lambda x: x.week).astype('int64')
test_flat['year'], test_flat['month'],test_flat['day'], test_flat['week']  = test_flat['datestr'].apply(lambda x: x.year).astype('int64'), test_flat['datestr'].apply(lambda x: x.month).astype('int64'), test_flat['datestr'].apply(lambda x: x.day).astype('int64'), test_flat['datestr'].apply(lambda x: x.week).astype('int64')
train_flat['day'].unique()
#The lineplot of the date columns

def plot_lineplot(train_flat, cols, col_y):
    for col in cols:
        fig = plt.figure(figsize=(15,8))
        sns.set_style("whitegrid")
        g = sns.lineplot(col, col_y, hue='device.isMobile', data=train_flat)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel('log of transaction revenue')# Set text for y axis
        fig.show()

col_y = train_flat['totals.transactionRevenueLogNAN']
cat_cols = ['datestr','day', 'month', 'year', 'week']   
plot_lineplot(train_flat, cat_cols, col_y)
def plot_box_mobile(train_flat, cols, col_y):
    for col in cols:
        fig = plt.figure(figsize=(20,8))
        sns.set_style("whitegrid")
        g = sns.boxplot(col, col_y, hue='device.isMobile', data=train_flat)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel('log of transaction revenue')# Set text for y axis
        for item in g.get_xticklabels():
            item.set_rotation(90)
        fig.show()

col_y = train_flat['totals.transactionRevenueLogNAN']
cat_cols = ['geoNetwork.continent','geoNetwork.subContinent','geoNetwork.metro', 'geoNetwork.city','trafficSource.source', 'trafficSource.medium']  
plot_box_mobile(train_flat, cat_cols, col_y)
# Device as a violin plot
def plot_violin(train_flat, cols, col_y):
    for col in cols:
        fig = plt.figure(figsize=(22,10))
        sns.set_style("whitegrid")
        g = sns.violinplot(col, col_y, data=train_flat)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel('log of transaction revenue')# Set text for y axis
        for item in g.get_xticklabels():
            item.set_rotation(90)
        fig.show()

col_y = train_flat['totals.transactionRevenueLogNAN']
cat_cols = ['device.isMobile','device.browser','device.deviceCategory','device.operatingSystem','trafficSource.adwordsClickInfo.isVideoAd']
#cat_cols = train_flat.select_dtypes(include='object')    
plot_violin(train_flat, cat_cols, col_y)
#The KDE of the numeric columns
def plot_jointplot(train_flat, cols, col_y):
    for col in cols:
        fig = plt.figure(figsize=(15,15))
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y , data=train_flat)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel('log of transaction revenue')# Set text for y axis
        fig.show()

col_y = train_flat['totals.transactionRevenueLogNAN']
cat_cols = ['totals.hits','visitNumber', 'totals.pageviews', 'totals.bounces', 'totals.newVisits', 'visitStartTime']
#cat_cols = train_flat.select_dtypes(include='object')    
plot_jointplot(train_flat, cat_cols, col_y)

##HITS and PAGEVIEWS have very similar distributions! We could probably drop one of them...., VISTNUMBER seems pretty different
#One more by all counties
def plot_box(train_flat, cols, col_y):
    for col in cols:
        fig = plt.figure(figsize=(150,25))
        sns.set_style("whitegrid")
        g = sns.boxplot(col, col_y, data=train_flat)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel('log of transaction revenue')# Set text for y axis
        for item in g.get_xticklabels():
            item.set_rotation(90)
        fig.show()

col_y = train_flat['totals.transactionRevenueLog']
cat_cols = [ 'geoNetwork.region']
#cat_cols = train_flat.select_dtypes(include='object')    
plot_box(train_flat, cat_cols, col_y)
#Step 3 - Cleanup/Drop those columns

##Column not in test set
train_flat.drop(columns='trafficSource.campaignCode', axis=1, inplace=True) ## Only for train

##Columns with only one value 'not available in demo dataset'
col_check = train_flat.loc[:,(train_flat == 'not available in demo dataset').any(axis=0)].columns

for col in col_check:
    if train_flat[col].nunique() <= 1:
        train_flat.drop(columns=col, axis=1, inplace=True)
        test_flat.drop(columns=col, axis=1, inplace=True)
        print(col, 'is dropped')
        
col_drop = [
    #Constant Values
    'socialEngagementType',
    
    ## (including in test on this run. Without these I get a 1.7681 LB score)
    ##'device.browser', 'device.deviceCategory', 'trafficSource.source', 'geoNetwork.metro',  'geoNetwork.city',
    'geoNetwork.networkDomain', 
    
    #Pageviews is too similar to page hits might be removed when training is capped. 'totals.pageviews'
    ##Too many NANs
    'trafficSource.adContent',
    'trafficSource.adwordsClickInfo.adNetworkType',
    'trafficSource.adwordsClickInfo.gclId',
    'trafficSource.keyword',
    'trafficSource.referralPath',
    'trafficSource.adwordsClickInfo.slot'    
    ]
        
for col in col_drop:
    train_flat.drop(columns=col, axis=1, inplace=True)
    test_flat.drop(columns=col, axis=1, inplace=True)
    print(col, 'is dropped')

print('All cleaned up')
# Impute 0 for missing target values
train_flat["totals.transactionRevenue"].fillna(0, inplace=True)
train_flat["totals.transactionRevenueLog"].fillna(0, inplace=True)
train_id = train_flat["fullVisitorId"].values
test_id = test_flat["fullVisitorId"].values

# label encode the categorical variables and convert the numerical variables to float
cat_cols = ['channelGrouping', 
            'device.operatingSystem', 
            #'geoNetwork.continent', 
            'geoNetwork.region', 
            'geoNetwork.metro',
            'geoNetwork.city',
            #'device.isMobile', 
            #'device.browser', 
            #'device.deviceCategory', 
            'trafficSource.source', 
            #'trafficSource.medium', 
            'day', 
            'month', 
            #'year', 
            'week', 
            #'totals.bounces', 
            'totals.newVisits'
           ]
for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_flat[col].values.astype('str')) + list(test_flat[col].values.astype('str')))
    train_flat[col] = lbl.transform(list(train_flat[col].values.astype('str')))
    test_flat[col] = lbl.transform(list(test_flat[col].values.astype('str')))
    
   # train_flat[col] = train_flat[col].astype('category')
    #test_flat[col] = test_flat[col].astype('category')

num_cols = ['totals.hits', 'visitNumber', 'visitStartTime', 'totals.pageviews']    
for col in num_cols:
    train_flat[col] = train_flat[col].astype(float)
    test_flat[col] = test_flat[col].astype(float)
print('Done with transformations!')
train_flat.info()
# Split the train dataset into development and valid based on time 
train_s1x = train_flat[train_flat['datestr']<='2017-06-30']
train_s2x = train_flat[train_flat['datestr']>'2017-06-30']
train_s1ylog = train_s1x["totals.transactionRevenueLog"].values
train_s2ylog = train_s2x["totals.transactionRevenueLog"].values

train_s1x = train_s1x[cat_cols + num_cols] 
train_s2x = train_s2x[cat_cols + num_cols] 
test_X = test_flat[cat_cols + num_cols] 

train_flat_x = train_flat[cat_cols + num_cols] 
train_flat_ylog = train_flat["totals.transactionRevenueLog"].values


train_s1x.info()
params = {"early_stopping_rounds":200, 
           "eval_metric" : 'rmse', 
            "eval_set" : [(train_s2x, train_s2ylog)],
           'eval_names': ['valid'],
           'verbose': 100,
          'feature_name': num_cols, # that's actually the default
         'categorical_feature': cat_cols # that's actually the default
         }
print('Start training...')
# train
gbm = lgb.LGBMRegressor(n_estimators=4000,                                         
                        learning_rate=0.017,            
                        num_leaves=68,            
                        metric= 'rmse',             
                        #max_bin=400,            
                        bagging_fraction=.8, #subsample            
                        feature_fraction=.8, #colsample_bytree            
                        bagging_frequency=10,            
                        bagging_seed=2018,            
                        max_depth=14,            
                        #reg_alpha=.2,            
                        #reg_lambda=.5,            
                        min_split_gain=.1,            
                        min_child_weight=.5,            
                        min_child_samples=300,            
                        silent=-1)
bst = gbm.fit(train_s1x, train_s1ylog, **params)
print('done')
gc.collect()
lgb.plot_importance(gbm,max_num_features=30)
#predictions = bst.predict(test_X, num_iteration=bst.best_iteration)
predictions = bst.predict(test_X)
submission = pd.DataFrame({ 'fullVisitorId': test_id,'PredictedLogRevenue': predictions })
submission = submission.groupby('fullVisitorId')['PredictedLogRevenue'].sum().reset_index()
submission.to_csv("GA_submission_LGBM_20180923_log1plimitedfeaturescatv2.csv", index=False)
#def rmse(y_true, y_pred):
    #return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

#scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
#scalarX.fit(train_s1x)

#X = scalarX.transform(train_s1x)


#model = Sequential()
#model.add(Dense(24,input_dim=10,activation='relu'))
#model.add(Dense(12,input_dim=10,activation='relu'))
#model.add(Dense(6))
#model.add(Dense(1))
#model.compile(optimizer='adam',loss='mse',metrics=[rmse])

#model.fit(X, train_s1ylog, epochs=5, verbose=0)
#history = model.fit(train_s1x, train_s1ylog,validation_data=(train_s2x, train_s2ylog), epochs=5,batch_size=100, verbose=2)
#plt.plot(history.history['rmse'])
#plt.show()
#preds = model.predict(test_X)