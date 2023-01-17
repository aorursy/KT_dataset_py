import os

import json

import numpy as np

import pandas as pd

from pandas.io.json import json_normalize

import random

import matplotlib.pyplot as plt

import multiprocessing as mp

import datetime as dt

import xgboost as xgb

import sklearn

import lightgbm as lgb

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import time

import sklearn.metrics

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot
#Determine Sizes of Training and Testing Sets

training_size = int(sum(1 for line in open('../input/train_v2.csv')))

testing_size = int(sum(1 for line in open('../input/test_v2.csv')))

print('Size of training set is: ', training_size, 'rows. Size of testing set is: ', testing_size, 'rows.')
#Load portion of training data to assess the data

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

df2 = pd.read_csv('../input/train_v2.csv', nrows=1000)

df2.head()
#Define function to parse the JSON entries

def Parser (df):

    'This function takes a Pandas dataframe as an input, parses the JSON columns in the dataframe, and produces a parsed Pandas dataframe'

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    for column in JSON_COLUMNS:

        column_as_df = json_normalize(df[column])

        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]

        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    return df
# Define function to randomly load in 1% of train/test data

def load_DF(filename):

    'This function takes a csv file as an input, and outputs 1% of that files rows randomly selected '

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource'] #JSon Columns in the training set

    num_lines = int(sum(1 for line in open(filename))) # Read number of file lines in the training dataset

    print('The number of rows in the file is ',num_lines) 

    n = int(num_lines-1) #number of records in file (excludes header)

    s = int(0.01*n) #Desired sample of 10 Percent of Dataset

    random.seed(25)

    skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list

    DF=pd.read_csv(filename, converters={column: json.loads for column in JSON_COLUMNS}, 

                        dtype={'fullVisitorId': 'str'},skiprows=skip)

    

    DF = Parser(DF)

    return DF
#Unparse Randomly chosen training data for feature engineering

trainingData = load_DF('../input/train_v2.csv')

#Unparse Randomly chosen testing data for feature engineering

testingData = load_DF('../input/test_v2.csv')
print(trainingData.shape)

trainingData.head()
trainingData.info()
print(testingData.shape)

testingData.head()
# Determine the number of columns in dataset with constant values 

columns_constant = [i for i in trainingData.columns if trainingData[i].nunique(dropna=False)==1 ]

columns_constant
# Analyze the missing values in the dataset 

def missing_values(data):

    'Input: Dataframe, Output: Dataframe. Function analyzes the missing values in a dataframe and returns the percentage missing.'

    #Count up all the null values 

    missing = data.isnull().sum().sort_values(ascending = False) 

    # Get Count of total values  

    Count= data.count().sort_values(ascending=False)

    #Express Missing values as a percent of total values 

    Percent = (missing/(Count+missing)*100).sort_values(ascending=False)

    df = pd.concat([missing, Count, Percent], axis=1, keys=['Missing', 'Count', 'Percent'])

    print (df[~(df['Percent'] == 0)]) # Returning values of nulls different of 0    

    return df
missing_values(trainingData)
missing_values(testingData)
datavis = trainingData.copy()

datavis.head()
datavis.info()
datavis["totals.transactionRevenue"] = datavis["totals.transactionRevenue"].astype('float')

gdf = datavis.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()



plt.figure(figsize=(8,6))

plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))

plt.xlabel('index', fontsize=12)

plt.ylabel('TransactionRevenue', fontsize=12)

plt.show()
nzi = pd.notnull(datavis["totals.transactionRevenue"]).sum()

paying = (gdf["totals.transactionRevenue"]>0).sum()

print("There are",paying,"paying customers")

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

print("Percentage of customers that generate revenue is",round(paying*100 / gdf.shape[0],2),"%")
def scatter_plot(cnt_srs, color):

    trace = go.Scatter(

        x=cnt_srs.index[::-1],

        y=cnt_srs.values[::-1],

        showlegend=False,

        marker=dict(

            color=color,

        ),

    )

    return trace

datavis['date'] = datavis['date'].apply(lambda x: dt.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))

cnt_srs = datavis.groupby('date')['totals.transactionRevenue'].agg(['size', 'count'])

cnt_srs.columns = ["count", "count of non-zero revenue"]

cnt_srs = cnt_srs.sort_index()

cnt_srs.index = cnt_srs.index.astype('str')

trace1 = scatter_plot(cnt_srs["count"], 'red')

trace2 = scatter_plot(cnt_srs["count of non-zero revenue"], 'blue')



fig = tools.make_subplots(rows=2, cols=1, vertical_spacing=0.08,

                          subplot_titles=["Date - Visitors", "Date - Paying Customers"])

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="Date Plots")

init_notebook_mode(connected=True)

iplot(fig, filename='date-plots')
def horizontal_bar_chart(cnt_srs, color):

    trace = go.Bar(

        y=cnt_srs.index[::-1],

        x=cnt_srs.values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



# Device Browser

cnt_srs = datavis.groupby('device.browser')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(149, 48, 90, 0.6)')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(149, 48, 90, 0.6)')

trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(149, 48, 90, 0.6)')



# Device Category

cnt_srs = datavis.groupby('device.deviceCategory')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace4 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(28, 151, 89, 0.6)')

trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(28, 151, 89, 0.6)')

trace6 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(28, 151, 89, 0.6)')



# Operating system

cnt_srs = datavis.groupby('device.operatingSystem')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace7 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(60, 150, 60, 0.6)')

trace8 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10),'rgba(60, 150, 60, 0.6)')

trace9 = horizontal_bar_chart(cnt_srs["mean"].head(10),'rgba(60, 150, 60, 0.6)')



# Creating two subplots

fig = tools.make_subplots(rows=3, cols=3, vertical_spacing=0.04, 

                          subplot_titles=["Device Browser - Visitors", "Device Browser - Paying Customers", "Device Browser - Mean Revenue",

                                          "Device Category - Visitors",  "Device Category - Paying Customers", "Device Category - Mean Revenue", 

                                          "Device OS - Visitors", "Device OS - Paying Customers", "Device OS - Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.append_trace(trace4, 2, 1)

fig.append_trace(trace5, 2, 2)

fig.append_trace(trace6, 2, 3)

fig.append_trace(trace7, 3, 1)

fig.append_trace(trace8, 3, 2)

fig.append_trace(trace9, 3, 3)



fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")

init_notebook_mode(connected=True)

iplot(fig, filename='device-plots')
# Continent

cnt_srs = datavis.groupby('geoNetwork.continent')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(78, 50, 120, 0.6)')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(78, 50, 120, 0.6)')

trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(78, 50, 120, 0.6)')



# Sub-continent

cnt_srs = datavis.groupby('geoNetwork.subContinent')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace4 = horizontal_bar_chart(cnt_srs["count"], 'black')

trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"], 'black')

trace6 = horizontal_bar_chart(cnt_srs["mean"], 'black')



# Network domain

cnt_srs = datavis.groupby('geoNetwork.networkDomain')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace7 = horizontal_bar_chart(cnt_srs["count"].head(10), 'orange')

trace8 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'orange')

trace9 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=3, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 

                          subplot_titles=["Continent - Visitors", "Continent - Paying Customers", "Continent - Mean Revenue",

                                          "Sub Continent - Visitors",  "Sub Continent - Paying Customers", "Sub Continent - Mean Revenue",

                                          "Network Domain - Visitors", "Network Domain - Paying Customers", "Network Domain - Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.append_trace(trace4, 2, 1)

fig.append_trace(trace5, 2, 2)

fig.append_trace(trace6, 2, 3)

fig.append_trace(trace7, 3, 1)

fig.append_trace(trace8, 3, 2)

fig.append_trace(trace9, 3, 3)



fig['layout'].update(height=1500, width=1200, paper_bgcolor='rgb(233,233,233)', title="Geography Plots")

init_notebook_mode(connected=True)

iplot(fig, filename='geo-plots')
cnt_srs = datavis.groupby('totals.pageviews')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(60), 'gray')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(60), 'gray')

trace5 = horizontal_bar_chart(cnt_srs["mean"].head(60), 'gray')



# Hits

cnt_srs = datavis.groupby('totals.hits')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", 'mean']

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace3 = horizontal_bar_chart(cnt_srs["count"].head(60), 'blue')

trace4 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(60), 'blue')

trace6 = horizontal_bar_chart(cnt_srs["mean"].head(60), 'blue')



# Creating two subplots

fig = tools.make_subplots(rows=2, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 

                          subplot_titles=["Total Pageviews - Visitors", "Total Pageviews - Paying Customers", "Total Pageviews - Mean Revenue",

                                          "Total Hits - Visitors",  "Total Hits - Paying Customers", "Total Hits - Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace5, 1, 3)

fig.append_trace(trace3, 2, 1)

fig.append_trace(trace4, 2, 2)

fig.append_trace(trace6, 2, 3)



fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Visitor Profile Plots")

init_notebook_mode(connected=True)

iplot(fig, filename='visitor-profile-plots')
cnt_srs = datavis.groupby('trafficSource.source')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'orange')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'orange')

trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'orange')



# Sub-continent

cnt_srs = datavis.groupby('trafficSource.medium')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace4 = horizontal_bar_chart(cnt_srs["count"], 'green')

trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"], 'green')

trace6 = horizontal_bar_chart(cnt_srs["mean"], 'green')



# Creating two subplots

fig = tools.make_subplots(rows=2, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 

                          subplot_titles=["Traffic Source - Visitors", "Traffic Source - Paying Customers", "Traffic Source - Mean Revenue",

                                          "Traffic Source Medium - Visitors",  "Traffic Source Medium - Paying Customers", "Traffic Source Medium - Mean Revenue"

                                          ])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.append_trace(trace4, 2, 1)

fig.append_trace(trace5, 2, 2)

fig.append_trace(trace6, 2, 3)



fig['layout'].update(height=1000, width=1200, paper_bgcolor='rgb(233,233,233)', title="Traffic Source Plots")

init_notebook_mode(connected=True)

iplot(fig, filename='traffic-source-plots')
cnt_srs = datavis.groupby('totals.sessionQualityDim')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(60), 'cyan')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(60), 'cyan')

trace5 = horizontal_bar_chart(cnt_srs["mean"].head(60), 'cyan')



# Creating two subplots

fig = tools.make_subplots(rows=2, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 

                          subplot_titles=["Session Quality - Visitors", "Session Quality - Paying Customers", "Session Quality - Mean Revenue",

                                          "Time On Site - Visitors",  "Time On Site - Paying Customers", "Time On Site - Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace5, 1, 3)

fig.append_trace(trace3, 2, 1)

fig.append_trace(trace4, 2, 2)

fig.append_trace(trace6, 2, 3)



fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Visitor Profile Plots")

init_notebook_mode(connected=True)

iplot(fig, filename='visitor-profile-plots')
def featureEngineering(df):

    'Function requires a dataframe as an input, and will output a feature Engineered dataframe'

    #Calculate number of constant columns that can be dropped

    columns_constant = [i for i in df.columns if df[i].nunique(dropna=False)==1 ]

    # Drop features that are very similar to our target feature or will have no analytical value

    similarToFeature = ['totals.totalTransactionRevenue'] + ['totals.transactions']

    noAnalyticalValue =  ['visitId']+['customDimensions']+['hits']+['trafficSource.adwordsClickInfo.adNetworkType']+['trafficSource.adwordsClickInfo.isVideoAd']+['trafficSource.adwordsClickInfo.page']+['trafficSource.adwordsClickInfo.slot']

    tot = similarToFeature + noAnalyticalValue

    if 'totals.totalTransactionRevenue' and 'totals.transactions' and 'visitId' and 'fullVisitorId' and 'customDimensions' and 'hits' and 'trafficSource.adwordsClickInfo.adNetworkType' and 'trafficSource.adwordsClickInfo.isVideoAd'and 'trafficSource.adwordsClickInfo.page' and 'trafficSource.adwordsClickInfo.slot'in df.columns:

        columns_drop = columns_constant + tot

        #Drop constant columns 

        df= df.drop(columns_drop, axis=1)

    else:

        df= df.drop(columns_constant, axis=1)

        

    #Encode the categoric features to numeric 

    categoric_features = ["channelGrouping", "device.browser", 

            "device.deviceCategory", "device.operatingSystem", 

            "geoNetwork.city", "geoNetwork.continent", 

            "geoNetwork.country", "geoNetwork.metro",

            "geoNetwork.networkDomain", "geoNetwork.region", 

            "geoNetwork.subContinent", "trafficSource.adContent", 

            "trafficSource.adwordsClickInfo.adNetworkType", 

            "trafficSource.adwordsClickInfo.gclId", 

            "trafficSource.adwordsClickInfo.page", 

            "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",

            "trafficSource.keyword", "trafficSource.medium", 

            "trafficSource.referralPath", "trafficSource.source",

            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect', 'device.isMobile']

    for i in categoric_features:

        if i in df.columns:

            encode = sklearn.preprocessing.LabelEncoder()

            encode.fit(df[i].values.astype('str'))

            df[i] = encode.transform(list(df[i].values.astype('str')))

    

    #Convert numeric features to float values

    numeric_features = ["totals.hits", "totals.pageviews",'totals.transactionRevenue',"visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits','totals.sessionQualityDim', 'totals.timeOnSite']

    for i in numeric_features:

        df[i] = df[i].astype(float)

    

    #Encode Zeroes for nan values in totals.bounces & totals.transactionRevenues

    df['totals.transactionRevenue'].fillna(0,inplace=True)

    

    #Convert date to datetime

    df['date'] = pd.to_datetime(df['date'], format = "%Y%m%d")

    return df
#FeatureEngineer training and testing data 

trainingData = featureEngineering(trainingData)

testingData = featureEngineering(testingData)
#Attempted Reading in Batches

# Function runs and produces an output but not as intended 

# def dataPipeline(filename):

#     'dataPipeline is a function that takes a csv as an input, and returns a fully pre-processed Pandas dataframe ready for'

#     'ML Algorithms'

#     #Start Time of Function

#     start_time = dt.datetime.now()

#     print("Started at ", start_time)

#     # Read number of file lines in the dataset

#     num_lines = int(sum(1 for line in open(filename))) 

#     # Process in chunk sizes of 1%

#     batchSize = int(0.01*num_lines)

#     # Initialize a List to hold all of the processed, chunked dataframes

#     result = []

#     #For Parser

#     JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

#     print('Entering into processing the csv file by batches')

#     for batch in pd.read_csv(filename, chunksize=batchSize,

#                      converters={column: json.loads for column in JSON_COLUMNS}, 

#                      dtype={'fullVisitorId': 'str'}):

#         parsed = Parser(batch)

#         processed = featureEngineering(parsed)

#         result.append(processed)

#     Output = pd.concat(result)

#     #Time it took for function to finish

#     end_time = dt.datetime.now()

#     elapsed_time = end_time - start_time

#     print ("Preprocessed ", filename, "completely. Elapsed time: ", elapsed_time)

#     return Output
print(trainingData.shape)

trainingData.head()
print(testingData.shape)

testingData.head()
#Split training set for validation set 

train_df = trainingData[trainingData['date']> dt.date(2016,11,30)]

validation_df = trainingData[trainingData['date']<=dt.date(2016,11,30)]



features = [] 

label = []

for i in trainingData.columns:

    if i != 'totals.transactionRevenue' and i!='date' and i!= 'fullVisitorId' and i!= 'visitStartTime':

        features.append(i)

    elif i == 'totals.transactionRevenue':

        label.append(i)

#Target

train_y = np.log1p(train_df[label])

validation_y = np.log1p(validation_df[label])

test_y = np.log1p(testingData[label])

#Features

train_x = train_df[features]

validation_x = validation_df[features]

test_x = testingData[features]
print('training dataframe shape:' , train_df.shape)

print('validation dataframe shape:' , validation_df.shape)

print('training features shape:' , train_x.shape)

print('validation features shape:' , validation_x.shape)

print('training label shape:' , train_y.shape)

print('validation label shape:' , validation_y.shape)

print('testing features shape:' , test_x.shape)

print('testing label shape:' , test_y.shape)
# Prior to early stopping, determine best hyperparameters for most important parameters in 

#LGBM models. n_estimators, one of the most important parameters, is determined by 

# early stopping, so a maximum limit is set here 

params_LGB ={'num_leaves': sp_randint(6, 50), 

             "learning_rate": sp_uniform(0.05, 0.4),

             'min_child_samples': sp_randint(100, 500), 

             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

             'subsample': sp_uniform(loc=0.2, scale=0.8), 

             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6)}



LGBReg = lgb.LGBMRegressor(n_estimators=1500, random_state=40, silent = True, n_jobs=4,max_depth =-1)

LGBHP_opt = RandomizedSearchCV(

    estimator=LGBReg, param_distributions=params_LGB, 

    n_iter=100,

    cv=3,scoring='neg_mean_squared_error',random_state=40, refit=True, verbose=True)
LGBHP_opt.fit(train_x,train_y)
FinalLGBParams = LGBHP_opt.best_params_

AdditionalParams = {'objective': 'regression', 'metric':'rmse', 'verbosity':-1}

for i in AdditionalParams:

    FinalLGBParams[i] = AdditionalParams[i]

FinalLGBParams
#Save the Optimum LGB Params to increase efficiency of not having to run each time to obtain them

SavedLGBParams = {'colsample_bytree': 0.8539721005885168,

 'learning_rate': 0.05722266438555726,

 'min_child_samples': 453,

 'min_child_weight': 1,

 'num_leaves': 35,

 'subsample': 0.6218338710274947,

 'objective': 'regression',

 'metric': 'rmse',

 'verbosity': -1}
#Define function to use Early Stopping LightGBM model with parameters determined from RandomizedSearchCV

def LGBM(train_x, train_y, val_x, val_y, test_x, test_y, LGBParams):

    lgbtrain = lgb.Dataset(train_x, label=train_y)

    lgbval = lgb.Dataset(val_x, val_y)

    

    start_time = time.time()

    model = lgb.train(LGBParams, lgbtrain, 1000, valid_sets = [lgbval], early_stopping_rounds=100, verbose_eval=100)

    end_time = time.time()

    print(end_time-start_time)

    

    predictions_test= model.predict(test_x, num_iteration = model.best_iteration)

    predictions_validation = model.predict(val_x, num_iteration= model.best_iteration)

    

    return predictions_test, predictions_validation, model
# Run LGBM

predictions_test, predictions_validation, modellgb = LGBM(train_x, train_y, validation_x, validation_y, test_x, test_y,SavedLGBParams)
#Testing Set

MSE_LGB_test = sklearn.metrics.mean_squared_error(test_y,predictions_test)

MAE_LGB_test= sklearn.metrics.mean_absolute_error(test_y, predictions_test)

RMSE_LGB_test = np.sqrt(MSE_LGB_test)

#Validation Set

MSE_LGB_val = sklearn.metrics.mean_squared_error(validation_y, predictions_validation)

MAE_LGB_val = sklearn.metrics.mean_absolute_error(validation_y, predictions_validation)

RMSE_LGB_val = np.sqrt(MSE_LGB_val)

print('Mean square Error for the LGBM testing set: {}, Mean absolute error for the LGBM testing set: {}, Root mean square error for the LGBM testing set: {}'.format(MSE_LGB_test, MAE_LGB_test, RMSE_LGB_test))



print('Mean square Error for the LGBM validation set: {}, Mean absolute error for the LGBM validation set: {}, Root mean square error for the LGBM validation set: {}'.format(MSE_LGB_val, MAE_LGB_val, RMSE_LGB_val))
#Obtain and plot figure importance for the LGBM

fig, ax = plt.subplots(figsize=(12,18))

lgb.plot_importance(modellgb, max_num_features=50, height=0.8, ax=ax)

ax.grid(False)

plt.title("LightGBM - Feature Importance", fontsize=15)

plt.show()
# Prior to early stopping, determine best hyperparameters for most important parameters in 

#XGB model. n_estimators, one of the most important parameters, is determined by 

# early stopping, so a maximum limit is set here 

params_XGB= {  

    "learning_rate": sp_uniform(0.05, 0.4),

    "colsample_bytree": sp_uniform(loc=0.4, scale=0.6),

    'max_depth': sp_randint(3, 40),

    "subsample": sp_uniform(loc=0.2, scale=0.8),

    "gamma": sp_uniform(0, 10),

    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

    "min_child_weight": [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

}



XGBReg = xgb.XGBRegressor(n_estimators=2500, random_state=40, silent = True, n_jobs=4)

XGBHP_opt = RandomizedSearchCV(

    estimator=XGBReg, param_distributions=params_XGB, 

    n_iter=50,cv=2,scoring='neg_mean_squared_error',random_state=40, refit=True, verbose=True)

XGBHP_opt.fit(train_x,train_y)
FinalXGBParams = XGBHP_opt.best_params_

FinalXGBParams
AdditionalParams = {'objective': 'reg:linear', 'eval_metric':'rmse', 'silent':True, 'random_state':40}

for i in AdditionalParams:

    FinalXGBParams[i] = AdditionalParams[i]

FinalXGBParams
#Save the Optimum XGB Params to increase efficiency of not having to run each time to obtain them

SavedXGBParams = {'colsample_bytree': 0.5454951393008803,

 'gamma': 4.834820481469037,

 'learning_rate': 0.050158252068833425,

 'max_depth': 36,

 'min_child_weight': 1,

 'reg_alpha': 100,

 'subsample': 0.9278827742521691,

 'objective': 'reg:linear',

 'eval_metric': 'rmse',

 'silent': True,

 'random_state': 40}

SavedXGBParams
#Define function to run XGBoost model

def XGB(train_x, train_y, val_x, val_y, test_x, test_y, paramsXGB):    

    xgb_train_dataMatrix = xgb.DMatrix(train_x, label = train_y)

    xgb_val_dataMatrix = xgb.DMatrix(val_x, label = val_y)

    xgb_test_dataMatrix = xgb.DMatrix(test_x, label= test_y)

    

    start_time = time.time()

    model = xgb.train(paramsXGB, xgb_train_dataMatrix, 

                      num_boost_round=2000, 

                      evals= [(xgb_train_dataMatrix, 'train'), (xgb_val_dataMatrix, 'valid')],

                      early_stopping_rounds=100, 

                      verbose_eval=100

                     )    

    end_time = time.time()

    print(end_time-start_time)

    

    y_pred_val = model.predict(xgb_val_dataMatrix, ntree_limit=model.best_ntree_limit)

    y_pred_test = model.predict(xgb_test_dataMatrix, ntree_limit=model.best_ntree_limit)

    return y_pred_test, y_pred_val, model
#Run XGBM

predictions_testingData, predictions_validationData, model = XGB(train_x, train_y, validation_x, validation_y, test_x, test_y,SavedXGBParams)
#Testing Set

MSE_XGB_testing = sklearn.metrics.mean_squared_error(test_y,predictions_testingData)

MAE_XGB_testing= sklearn.metrics.mean_absolute_error(test_y, predictions_testingData)

RMSE_XGB_testing = np.sqrt(MSE_XGB_testing)



#Validation Set

MSE_XGB_validation = sklearn.metrics.mean_squared_error(validation_y, predictions_validationData)

MAE_XGB_validation = sklearn.metrics.mean_absolute_error(validation_y, predictions_validationData)

RMSE_XGB_validation = np.sqrt(MSE_XGB_validation)

print('Mean square Error for the XGB testing set: {}, Mean absolute error for the XGB testing set: {}, Root mean square error for the XGB testing set: {}'.format(MSE_XGB_testing, MAE_XGB_testing, RMSE_XGB_testing))



print('Mean square Error for the XGB validation set: {}, Mean absolute error for the XGB validation set: {}, Root mean square error for the XGB validation set: {}'.format(MSE_XGB_validation, MAE_XGB_validation, RMSE_XGB_validation))
#Obtain and plot figure importance for the LGBM

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

ax.grid(False)

plt.title("XGB - Feature Importance", fontsize=15)

plt.show()