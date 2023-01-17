# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from kaggle.competitions import twosigmanews
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
news_train_df = news_train_df.iloc[955479:1255479,]
#real_market = market_train_df
for a in range(0,len(stock_list)):
    substring = stock_list[a]
    asset_check = market_train_df.loc[market_train_df['assetCode'].str.contains(substring)]
  #  if any(substring in asset_check for substring in stock_list):
    real_market = real_market.append(asset_check,sort=False)
    print(a)
real_news = pd.DataFrame(columns=news_train_df.columns.values)
for b in range(0,len(stock_list)):
    substring = stock_list[b]
    asset_check = news_train_df.loc[news_train_df['assetCodes'].str.contains(substring)]
  #  if any(substring in asset_check for substring in stock_list):
    real_news = real_news.append(asset_check,sort=False)
    print(b)
market_train_df = real_market
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
manipulate = news_train_df
manipulate = manipulate.drop(['time','sourceTimestamp','firstCreated','sourceId','headline','subjects','audiences','headlineTag','marketCommentary','assetCodes','assetName','provider'],axis=1)
SC = KMeans(n_clusters=3)
print('k means done')
train = manipulate
std_scale = StandardScaler()
train = std_scale.fit_transform(train)
train_predict = SC.fit_predict(train)
train_trans = SC.transform(train)
principal = PCA(n_components=6)
train_pca = principal.fit_transform(train)
plt.scatter(train_pca[:,0], train_pca[:,1], c=SC.labels_.astype(float))
plt.xlabel('First PCA')
plt.ylabel('Second PCA')
plt.title('Training Data 2 Dimensional PCA With Cluster Coloring')
news_train_df['time'] = news_train_df['time'].apply(lambda x: date.toordinal(x))
news_test_df['time'] = news_test_df['time'].apply(lambda x: date.toordinal(x))
market_train_df = market_train_df.loc[market_train_df['time'] <= np.max(news_train_df['time'].astype(float).values.reshape(-1,1))]
market_train_df = market_train_df.loc[market_train_df['time'] >= np.min(news_train_df['time'].astype(float).values.reshape(-1,1))]
col_list = list(market_train_df.columns.values)
col_list.append('PCA1')
col_list.append('PCA2')
col_list.append('PCA3')
col_list.append('PCA4')
col_list.append('PCA5')
col_list.append('PCA6')
news_train_df = news_train_df.assign(PCA1=train_pca[:,0])
news_train_df = news_train_df.assign(PCA2=train_pca[:,1])
news_train_df = news_train_df.assign(PCA3=train_pca[:,2])
news_train_df = news_train_df.assign(PCA4=train_pca[:,3])
news_train_df = news_train_df.assign(PCA5=train_pca[:,4])
news_train_df = news_train_df.assign(PCA6=train_pca[:,5])
returns_test = pd.DataFrame(columns=market_train_df.columns.values)
returns_train = pd.DataFrame(columns=market_train_df.columns.values)
size = 0
unique_dates = news_train_df['time'].unique()
for i in range(0,len(unique_dates)):
    date = unique_dates[i]
    row_news = news_train_df.loc[news_train_df['time']==date,:]
    market_rows = market_train_df.loc[(market_train_df['time']==date),:]
    subjects = row_news['assetCodes']
    pca1 = row_news['PCA1']
    pca2 = row_news['PCA2']
    pca3 = row_news['PCA3']
    pca4 = row_news['PCA4']
    pca5 = row_news['PCA5']
    pca6 = row_news['PCA6']
    for j in range(0,len(subjects)):
        subject = subjects.iloc[j].split()[0]
        train_pca1 = pca1.iloc[j]
        train_pca2 = pca2.iloc[j]
        train_pca3 = pca3.iloc[j]
        train_pca4 = pca4.iloc[j]
        train_pca5 = pca5.iloc[j]
        train_pca6 = pca6.iloc[j]
        subject_start = subject.find("'")
        subject_end = subject.find(".")
        subject = subject[(subject_start+1):subject_end]
        market_row = market_rows.loc[market_rows['assetCode'].str.contains(subject)]
        if (market_row.shape[0] != 0):
            market_row.insert(0, 'PCA1', train_pca1)
            market_row.insert(0, 'PCA2', train_pca2)
            market_row.insert(0, 'PCA3', train_pca3)
            market_row.insert(0, 'PCA4', train_pca4)
            market_row.insert(0, 'PCA5', train_pca5)
            market_row.insert(0, 'PCA6', train_pca6)
            returns_train = returns_train.append(market_row,sort=False)
    print(i)
print('training done')
manipulate = news_test_df
manipulate = manipulate.drop(['time','sourceTimestamp','firstCreated','sourceId','headline','subjects','audiences','headlineTag','marketCommentary','assetCodes','assetName','provider'],axis=1)
test = manipulate
test = std_scale.fit_transform(test)
test_predict = SC.predict(test)
test_trans = SC.transform(test)
test_pca = principal.transform(test)
col_list = list(market_train_df.columns.values)
col_list.append('PCA1')
col_list.append('PCA2')
col_list.append('PCA3')
col_list.append('PCA4')
col_list.append('PCA5')
col_list.append('PCA6')
news_test_df = news_test_df.assign(PCA1=test_pca[:,0])
news_test_df = news_test_df.assign(PCA2=test_pca[:,1])
news_test_df = news_test_df.assign(PCA3=test_pca[:,2])
news_test_df = news_test_df.assign(PCA4=test_pca[:,3])
news_test_df = news_test_df.assign(PCA5=test_pca[:,4])
news_test_df = news_test_df.assign(PCA6=test_pca[:,5])
size = 0
unique_dates = news_test_df['time'].unique()
for i in range(0,len(unique_dates)):
    date = unique_dates[i]
    row_news = news_test_df.loc[news_test_df['time']==date,:]
    market_rows = market_train_df.loc[(market_train_df['time']==date),:]
    subjects = row_news['assetCodes']
    pca1 = row_news['PCA1']
    pca2 = row_news['PCA2']
    pca3 = row_news['PCA3']
    pca4 = row_news['PCA4']
    pca5 = row_news['PCA5']
    pca6 = row_news['PCA6']
    for j in range(0,len(subjects)):
        subject = subjects.iloc[j].split()[0]
        train_pca1 = pca1.iloc[j]
        train_pca2 = pca2.iloc[j]
        train_pca3 = pca3.iloc[j]
        train_pca4 = pca4.iloc[j]
        train_pca5 = pca5.iloc[j]
        train_pca6 = pca6.iloc[j]
        subject_start = subject.find("'")
        subject_end = subject.find(".")
        subject = subject[(subject_start+1):subject_end]
        market_row = market_rows.loc[market_rows['assetCode'].str.contains(subject)]
        if (market_row.shape[0] != 0):
            market_row.insert(0, 'PCA1', train_pca1)
            market_row.insert(0, 'PCA2', train_pca2)
            market_row.insert(0, 'PCA3', train_pca3)
            market_row.insert(0, 'PCA4', train_pca4)
            market_row.insert(0, 'PCA5', train_pca5)
            market_row.insert(0, 'PCA6', train_pca6)
            returns_test = returns_test.append(market_row,sort=False)
    print(i)
returns_train = returns_train.reset_index()
returns_test = returns_test.reset_index()
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
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA

def function(x_test,x_train,y_test,y_train,model):
        if(model == "linear"):
                lin = LinearRegression()
                lin.fit(x_train,y_train)
                return lin.predict(x_test)
        elif(model == "logistic"):
                logit = LogisticRegression()
                logit.fit(x_train,y_train)
                return logit.predict(x_test)
        elif(model == "svc"):
                model3 = SVC(gamma='scale')
                model3.fit(x_train,y_train)
                return model3.predict(x_test)
        elif(model == "svr"):
                model3 = SVR(gamma='scale')
                model3.fit(x_train,y_train)
                return model3.predict(x_test)
        elif(model == "lda"):
                model4 = LDA()
                model4.fit(x_train,y_train)
                return model4.predict(x_test)
        elif(model == "qda"):
                model5 = QDA(reg_param=0.05)
                model5.fit(x_train,y_train)
                return model5.predict(x_test)
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
returns_test = returns_test.dropna()
returns_train = returns_train.dropna()
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(returns_train['returnsOpenNextMktres10'])
lab_enc = preprocessing.LabelEncoder()
y_test = lab_enc.fit_transform(returns_test['returnsOpenNextMktres10'])
y_train_nolab = returns_train['returnsOpenNextMktres10']
y_test_nolab = returns_train['returnsOpenNextMktres10']
y_train = y_train_nolab*100000
y_train = y_train.astype('int')
y_test = y_test_nolab*100000
y_test = y_test.astype('int')
lm_results_nolab = function(returns_test.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),returns_train.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),y_test_nolab,y_train_nolab,"linear")
lm_results = function(returns_test.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),returns_train.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),y_test,y_train,"linear")
lg_results = function(returns_test.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),returns_train.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),y_test,y_train,"logistic")
svm_results = function(returns_test.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),returns_train.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),y_test,y_train,"svc")
svr_results = function(returns_test.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),returns_train.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),y_test,y_train,"svr")
lda_results = function(returns_test.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),returns_train.drop(['returnsOpenNextMktres10','time','assetCode','assetName'],axis=1),y_test,y_train,"lda")
lm_results = scaler.fit_transform(lm_results.reshape(-1, 1))
lg_results = scaler.fit_transform(lg_results.reshape(-1, 1))
svm_results = scaler.fit_transform(svm_results.reshape(-1, 1))
svr_results = scaler.fit_transform(svr_results.reshape(-1, 1))
lda_results = scaler.fit_transform(lda_results.reshape(-1, 1))
lm_mse_nolab = mean_squared_error(returns_test['returnsOpenNextMktres10'],lm_results_nolab)
lm_mse = mean_squared_error(returns_test['returnsOpenNextMktres10'],lm_results)
lg_mse = mean_squared_error(returns_test['returnsOpenNextMktres10'],lg_results)
svm_mse = mean_squared_error(returns_test['returnsOpenNextMktres10'],svm_results)
svr_mse = mean_squared_error(returns_test['returnsOpenNextMktres10'],svr_results)
lda_mse = mean_squared_error(returns_test['returnsOpenNextMktres10'],lda_results)
svm_plot = returns_test['returnsOpenNextMktres10'].astype(float).values.reshape(-1,1) - svm_results
plt.plot(svm_plot[0:1000])
plt.xlabel('Test Instance Number')
plt.ylabel('Real - SVM Prediction')
plt.title('SVM Prediction Compared to Real')
list_mse = [lm_mse,lg_mse,svm_mse,lda_mse]


plt.bar(x=[0,10,20,30,40,50],height=[lm_mse,lm_mse_nolab,lg_mse,svm_mse,svr_mse,lda_mse],tick_label=['LM MSE','LM NO LAB MSE','LG MSE','SVC MSE','SVR MSE','LDA MSE'])
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('MSE of Various Models')
lda_plot = returns_test['returnsOpenNextMktres10'].astype(float).values.reshape(-1,1) - lda_results
plt.plot(lda_plot[0:1000])
plt.xlabel('Test Instance Number')
plt.ylabel('Real - SVM Prediction')
plt.title('LDA Prediction Compared to Real')
returns_test = returns_test.assign(LM=lm_results)
returns_test = returns_test.assign(LM_NoLab = lm_results_nolab)
returns_test = returns_test.assign(LG = lg_results)
returns_test = returns_test.assign(SVC = svm_results)
returns_test = returns_test.assign(SVR = svr_results)
returns_test = returns_test.assign(LDA = lda_results)
lm_list = []
lm_list_nolab = []
lg_list = []
svm_list = []
svr_list = []
lda_list = []
unique_dates = returns_test['time'].unique()
for i in range(0,len(unique_dates)):
    date = unique_dates[i]
    rows = returns_test.loc[returns_test['time']==date,]
    lm_list.append(rows['LM'].multiply(rows['returnsOpenPrevMktres10']).sum())
    lg_list.append(rows['LG'].multiply(rows['returnsOpenPrevMktres10']).sum())
    lm_list_nolab.append(rows['LM_NoLab'].multiply(rows['returnsOpenPrevMktres10']).sum())
    svm_list.append(rows['SVC'].multiply(rows['returnsOpenPrevMktres10']).sum())
    svr_list.append(rows['SVR'].multiply(rows['returnsOpenPrevMktres10']).sum())
    lda_list.append(rows['LDA'].multiply(rows['returnsOpenPrevMktres10']).sum())
lm_score = np.mean(lm_list)/np.std(lm_list)
lm_nolab_score = np.mean(lm_list)/np.std(lm_list_nolab)  
lg_score = np.mean(lg_list)/np.std(lg_list)  
svm_score = np.mean(svm_list)/np.std(svm_list)  
svr_score = np.mean(svr_list)/np.std(svr_list)  
lda_score = np.mean(lda_list)/np.std(lda_list)