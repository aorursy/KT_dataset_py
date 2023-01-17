# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from kaggle.competitions import twosigmanews
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import date
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
env = twosigmanews.make_env()
stock_list = ['MSFT','AAPL','AMZN','BRK.B','JNJ','JPM','FB','GOOG','XOM','PFE','AMD', 'FLS', 'HCA', 'PKI', 'ILMN', 'GLW', 'IQV', 'WCG', 'VRTX', 'QCOM', 'ARNC', 'UAL', 'NKTR', 'LLY', 'ORLY', 'RCL', 'BLL', 'CA', 'AAP', 'XLNX', 'CF', 'LUV', 'WBA','CI','ESRX']
(market_train_df, news_train_df) = env.get_training_data()
real_market = pd.DataFrame(columns=market_train_df.columns.values)
real_market = market_train_df
#for a in range(0,10000):
#    row_check = market_train_df.iloc[a,:]
 #   asset_check = row_check.loc['assetCode']
  #  if any(substring in asset_check for substring in stock_list):
  #      real_market = real_market.append(row_check,sort=False)
#real_news = pd.DataFrame(columns=news_train_df.columns.values)
real_news = news_train_df.iloc[9554579:10554578,]
market_train_df['time'] = market_train_df['time'].apply(lambda x: date.toordinal(x))
print('news done')
provider_keys = []
providerInt = []        
for x in real_news['provider']:
    if(x not in provider_keys):
        provider_keys.append(x)
    providerInt.append(provider_keys.index(x))
real_news.insert(35, 'ProviderInt', providerInt)
print('provider done')
news_train_df, news_test_df = train_test_split(real_news)
market_train_df = real_market
manipulate = news_train_df
manipulate = manipulate.drop(['time','sourceTimestamp','firstCreated','sourceId','headline','subjects','audiences','headlineTag','marketCommentary','assetCodes','assetName','provider'],axis=1)
SC = KMeans(n_clusters=3)
print('k means done')
train = manipulate
train_predict = SC.fit_predict(train)
train_trans = SC.transform(train)
principal = PCA(n_components=2)
train_pca = principal.fit_transform(train)
plt.scatter(train_pca[:,0], train_pca[:,1], c=SC.labels_.astype(float))
plt.xlabel('First PCA')
plt.ylabel('Second PCA')
plt.title('Training Data 2 Dimensional PCA With Cluster Coloring')
market_train_df['time'] = market_train_df['time'].apply(lambda x: date.toordinal(x))
market_train_df = market_train_df.loc[market_train_df['time'] <= np.mean(news_train_df.loc['time'].astype(float).values.reshape(-1,1))]
col_list = list(market_train_df.columns.values)
col_list.append('PCA1')
col_list.append('PCA2')
returns_test = pd.DataFrame(columns=market_train_df.columns.values)
returns_train = pd.DataFrame(columns=market_train_df.columns.values)
size = 0
for i in range(0,train.shape[0]):
    row_news = news_train_df.iloc[i,:]
    date_news = date.toordinal(row_news['time'])
    subject_news = row_news['assetCodes']
    subject_news = subject_news.split()
    for j in range(0,len(subject_news)):
        subject = subject_news[j]
        subject = subject.replace("{",'')
        subject = subject.replace("'",'')
        subject = subject.replace("}",'')
        subject = subject.replace(",",'')
        market_rows = market_train_df.loc[(market_train_df['time']==date_news),:]
        if (market_rows.shape[0] != 0):
                market_row = market_rows.loc[(market_train_df['assetCode']==subject),:]
                if (market_row.shape[0] != 0):
                    train_cluster = train_predict[i]
                    market_row.insert(0, 'PCA1', train_pca[i,0])
                    market_row.insert(0, 'PCA2', train_pca[i,1])
                    returns_train = returns_train.append(market_row,sort=False)
                    size = size + 1
print('training done')
manipulate = news_test_df
manipulate = manipulate.drop(['time','sourceTimestamp','firstCreated','sourceId','headline','subjects','audiences','headlineTag','marketCommentary','assetCodes','assetName','provider'],axis=1)
test = manipulate
test_predict = SC.predict(test)
test_trans = SC.transform(test)
test_pca = principal.transform(test)
col_list = list(market_train_df.columns.values)
col_list.append('PCA1')
col_list.append('PCA2')
size = 0
for i in range(0,test.shape[0]):
    row_news = news_test_df.iloc[i,:]
    date_news = date.toordinal(row_news['time'])
    subject_news = row_news['assetCodes']
    subject_news = subject_news.split()
    for j in range(0,len(subject_news)):
        subject = subject_news[j]
        subject = subject.replace("{",'')
        subject = subject.replace("'",'')
        subject = subject.replace("}",'')
        subject = subject.replace(",",'')
        market_rows = market_train_df.loc[(market_train_df['time']==date_news),:]
        if (market_rows.shape[0] != 0):
                market_row = market_rows.loc[(market_train_df['assetCode']==subject),:]
                if (market_row.shape[0] != 0):
                    test_cluster = test_predict[i]
                    market_row.insert(0, 'PCA1', test_pca[i,0])
                    market_row.insert(0, 'PCA2', test_pca[i,1])
                    returns_test = returns_test.append(market_row,sort=False)
                    size = size + 1
scaler = MinMaxScaler(feature_range=(-1,1))  
returns_train['returnsOpenNextMktres10'] = scaler.fit_transform(returns_train['returnsOpenNextMktres10'].astype(float).values.reshape(-1, 1))
returns_test['returnsOpenNextMktres10'] = scaler.fit_transform(returns_test['returnsOpenNextMktres10'].astype(float).values.reshape(-1, 1))
# Any results you write to the current directory are saved as output.
returns_train.to_csv('returns_train.csv',index=False)
returns_test.to_csv('returns_test.csv',index=False)
market_train_df.to_csv('market_train_df.csv',index=False)
news_train_df.to_csv('news_train_df.csv',index=False)
news_test_df.to_csv('news_test_df.csv',index=False)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR

def function(x_test,x_train,y_test,y_train,model):
        if(model == "linear"):
                lin = LinearRegression()
                lin.fit(x_train,y_train)
                return lin.predict(x_test)
        elif(model == "logistic"):
                logit = LogisticRegression()
                logit.fit(x_train,y_train)
                return logit.predict(x_test)
        elif(model == "svm"):
                model3 = SVC(gamma='scale')
                model3.fit(x_train,y_train)
                return model3.predict(x_test)
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(returns_train['returnsOpenNextMktres10'])
lab_enc = preprocessing.LabelEncoder()
y_test = lab_enc.fit_transform(returns_test['returnsOpenNextMktres10'])
lm_results = function(returns_test.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),returns_train.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),y_test,y_train,"linear")
lg_results = function(returns_test.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),returns_train.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),y_test,y_train,"logistic")
svm_results = function(returns_test.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),returns_train.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),y_test,y_train,"svm")
lm_results = scaler.fit_transform(lm_results.reshape(-1, 1))
lg_results = scaler.fit_transform(lg_results.reshape(-1, 1))
svm_results = scaler.fit_transform(svm_results.reshape(-1, 1))
lm_mse = mean_squared_error(returns_test['returnsOpenNextMktres10'],lm_results)
lg_mse = mean_squared_error(returns_test['returnsOpenNextMktres10'],lg_results)
svm_mse = mean_squared_error(returns_test['returnsOpenNextMktres10'],svm_results)
svm_plot = returns_test['returnsOpenNextMktres10'].astype(float).values.reshape(-1,1) - svm_results
plt.plot(svm_plot)
plt.xlabel('Test Instance Number')
plt.ylabel('Real - SVM Prediction')
plt.title('SVM Prediction Compared to Real')
list_mse = [lm_mse,lg_mse,svm_mse]

plt.bar(x=[1,2,3,],height=[lm_mse,lg_mse,svm_mse],tick_label=['LM MSE','LG MSE','SVM MSE'])
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('MSE of Various Models')