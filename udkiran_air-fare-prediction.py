# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import datasets, linear_model
from scipy import linalg
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import requests
cols = ['City1', 'City2', 'Average Fare', 'Distance', 'Average weekly passengers',
                   'market leading airline', 'market share', 'Average fare', 'Low price airline',
                   'market share', 'price']
target_url = 'http://users.stat.ufl.edu/~winner/data/airq402.dat'

response = requests.get(target_url)
data_dat = response.text

data_list = data_dat.splitlines() 

data_content = [e.split() for e in data_list] 
data = pd.DataFrame(data_content, columns = cols)
data.head()
data.columns = data.columns.str.replace(" ","_")
data.columns = data.columns.str.replace(".","_")
data.head()
data.rename(columns={data.columns[7] : 'Average_fare_1'}, inplace=True)
data.head()

column_names = data.columns.values
column_names[9] = 'market_share_1'
data.columns = column_names

data.head()

# Converting string of numbers (as a result of .split()) to floats

data['Average_Fare'] = data['Average_Fare'].astype(float)
data['Distance'] = data['Distance'].astype(float)
data['Average_weekly_passengers'] = data['Average_weekly_passengers'].astype(float)
data['market_share'] = data['market_share'].astype(float)
data['Average_fare_1'] = data['Average_fare_1'].astype(float)
data['market_share_1'] = data['market_share_1'].astype(float)
data['price'] = data['price'].astype(float)

#Since there is not much difference between Average_Fare and Average_Fare_1
data1 = data.drop(['Average_fare_1'], axis = 1)
data1.head()

data1.describe()

data1.describe(include= ['O'])

data1.isnull().any()

## EDA

print(data1['Average_Fare'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(data1['Average_Fare'], color='b', bins=20, hist_kws={'alpha': 0.4});

#skewed towards the right , with some outliers

data1_num = data1.select_dtypes(include = ['float64', 'int64'])
data1_num.head()

### Numerical feature analysis:

data1_num_corr = data1_num.corr()['Average_Fare'][:-1] # -1 because the latest row is Average_Fare
golden_features_list = data1_num_corr[abs(data1_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with Average_Fare:\n{}".format(len(golden_features_list), golden_features_list))

#Plotting scatter plots of Independent Variable vs Dependent Variable.

for i in range(0, len(data1_num.columns), 6):
    sns.pairplot(data=data1_num,
                x_vars=data1_num.columns[i:i+5],
                y_vars=['Average_Fare'])

# we can see that out of these numerical features, distance, price, market_share share some relationship with the target variable

# Average_weekly_passengers - data is scattered with huge outliers, and most data is concentrated within sum of 2500. Very random, no pattern

# market_share1 seems to be uniformly distributed.

corr = data1_num.drop('Average_Fare', axis=1).corr() # Check for multicollinearity using heat map
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap="PiYG", vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);

# the correlation between distance and price is highest 0.58
# the correlation between distance and market_share is 2nd highest -0.53
# the market_share and price however share a low correlation of -0.31 (so we will keep them both)
# market_share_1 has a higher correlation 0.37 (>0.31) with distance as compared to market_share but still it is less
# average_weekly_passengers doesn't have correlation with any of the features nor with the target variable

### Categorical feature analysis:

data1_cat = data1.select_dtypes(include = ['object'])
data1_cat.head()

x=data1.groupby(['City1', 'Average_Fare'], as_index = False).count().sort_values(by = 'Average_Fare',ascending = False)
x.head()

sns.barplot('City1', 'Average_Fare', data=np.round(data1,3), color="blue")
plt.show()

sns.barplot('City2', 'Average_Fare', data=np.round(data1,3), color="blue")
plt.show()

sns.barplot('market_leading_airline', 'Average_Fare', data=np.round(data1,3), color="blue")
plt.show()

sns.barplot('Low_price_airline', 'Average_Fare', data=np.round(data1,3), color="blue")
plt.show()

#Categorical variables have too many variables for City1 and City2, encoding is not an option for them

data1.head()

## Check for outliers

#Distance
import seaborn as sns
 
sns.boxplot(data=data1.iloc[:,3], color="blue")

#Average weekly pass
import seaborn as sns
 
sns.boxplot(data=data1.iloc[:,4], color="blue")

# huge no. of outliers

#market_share
import seaborn as sns
 
sns.boxplot(data=data1.iloc[:,6], color="blue") 

## market_share1
import seaborn as sns
 
sns.boxplot(data=data1.iloc[:,8], color="blue")  

#outlier treatment -  clubbing beyond P1 and P99 percentiles

dist = data1_num['Distance'].values  #series to array 
p1_dist = np.percentile(dist, 1) #percentile values
p2_dist = np.percentile(dist, 99)
print(p1_dist, p2_dist)

passe = data1_num['Average_weekly_passengers'].values  #series to array 
p1_passe = np.percentile(passe, 1) #percentile values
p2_passe = np.percentile(passe, 99)
print(p1_passe, p2_passe)

ms = data1_num['market_share'].values  #series to array 
p1_ms = np.percentile(ms, 1) #percentile values
p2_ms = np.percentile(ms, 99)
print(p1_ms, p2_ms)

ms1 = data1_num['market_share_1'].values  #series to array 
p1_ms1 = np.percentile(ms1, 1) #percentile values
p2_ms1 = np.percentile(ms1, 99)
print(p1_ms1, p2_ms1)

data1_list = [data1]

for dataset1 in data1_list:
    dataset1.loc[dataset1.Distance < 187.95, 'Distance' ] = 187.95
    dataset1.loc[dataset1.Distance > 2586.01, 'Distance' ] = 2586.01
    
    dataset1.loc[dataset1.Average_weekly_passengers < 184.33, 'Average_weekly_passengers' ] = 184.33
    dataset1.loc[dataset1.Average_weekly_passengers > 3699.30, 'Average_weekly_passengers' ] = 3699.30
    
    dataset1.loc[dataset1.market_share < 22.9443, 'market_share' ] = 22.9443 
    dataset1.loc[dataset1.market_share > 99.6801, 'market_share' ] = 99.6801
    
    dataset1.loc[dataset1.market_share_1 < 1.3099, 'market_share_1' ] = 1.3099
    dataset1.loc[dataset1.market_share_1 > 99.6801, 'market_share_1' ] = 99.6801

## Dummy treatment for categorical data

df1 = data1
df1.head()

MLA_Dum = pd.get_dummies(df1.market_leading_airline,prefix='MLA',drop_first=True)
MLA_Dum.head()

LPA_Dum = pd.get_dummies(df1.Low_price_airline,prefix='LPA',drop_first=True)
LPA_Dum.head()

modeldata = pd.concat([df1, MLA_Dum, LPA_Dum],axis=1)
modeldata.head()

modeldata1 = modeldata.drop(['City1', 'City2', 'market_leading_airline', 'Low_price_airline'], axis = 1)
modeldata1.head()

## Cross Validation

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(modeldata1,test_size=0.30,random_state=1234)

df_train.shape, df_test.shape

XTrain = df_train.iloc[:,list(range(1,6)) + list(range(6,df_train.shape[1]))]
YTrain = df_train['Average_Fare']

XTest = df_test.iloc[:,list(range(1,6)) + list(range(6,df_test.shape[1]))]
YTest = df_test['Average_Fare']

XTrain.shape,YTrain.shape

XTest.shape,YTest.shape

## Multiple Linear Regression Model

regr = linear_model.LinearRegression(normalize=True)
regr

regr.fit(XTrain,YTrain)

regr.score(XTrain,YTrain)

regr.score(XTest,YTest)

from scipy import stats

def acc(X, Y, model):
    Y = np.array(Y)
    yhat = model.predict(X)
    SSR = sum((Y-yhat)**2)
    SST = sum((Y - np.mean(Y))**2)
    Rsquared = 1 - (float(SSR))/SST
    adjRsquared = 1 - (1-Rsquared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
    
    return Rsquared, adjRsquared

acc(XTest, YTest, regr)

## Feature importance

def ErrorMetric(model,X,Y):
    Yhat = model.predict(X)
    MAPE = np.mean(abs(Y-Yhat)/Y)*100
    MSSE = np.mean(np.square(Y-Yhat))
    return MAPE, MSSE

# for varible importance, use ensemble method

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
model = ensemble.GradientBoostingRegressor()
model.fit(XTrain, YTrain)

# Score for XGB regressor

print('Gradient Boosting score": %.4f' % model.score(XTest, YTest))

feature_labels = np.array(modeldata1.columns)
feature_labels

importance = model.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))

s1 = pd.Series(list(feature_labels), name='feature')

s2 = pd.Series(list(importance), name='variable imp')

imp = pd.concat([s1, s2], axis=1)
imp.head()

imp_df = imp[imp['variable imp'] > .02] 
imp_df.head(11)

from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(15,8))
ax = sns.barplot('variable imp', 'feature', data=np.round(imp_df,3), color="green")

## PCA

modeldata1.head()

features = list(feature_labels)

# Separating out the features
x= modeldata1.loc[:, features].values

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf1 = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2', 'pc3'])

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())

principalDf1.head()

result = principalDf1
# Run The PCA
sns.set_style("white")
# Store results of PCA in a data frame
#result=pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(3)], index=df.index)
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
modeldata1['Average_Fare']=pd.Categorical(modeldata1['Average_Fare'])
my_color=modeldata1['Average_Fare'].cat.codes

# Plot initialisation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result['pc1'], result['pc2'], result['pc3'], c=my_color, cmap='viridis', s=60)
 
# make simple, bare axis lines through space:
xAxisLine = ((min(result['pc1']), max(result['pc1'])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'black')
yAxisLine = ((0, 0), (min(result['pc2']), max(result['pc2'])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'black')
zAxisLine = ((0, 0), (0,0), (min(result['pc3']), max(result['pc3'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'black')
 
# label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA on the train data")

pca = PCA(n_components=9)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8', 'pc9'])

print (pca.explained_variance_ratio_.cumsum())

# so optimal number of PC components = 7

pca = PCA(n_components=7)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7'])

principalDf.head()

#from sklearn.cross_validation import train_test_split
#Join 'default' in to the train_pDf dataframe
train_principalDf_xy = pd.concat([principalDf, modeldata1['Average_Fare']], axis = 1) 

train_principalDf_xy_1 = train_principalDf_xy

#train_principal_DF_xy_1 is the final transformed train data set

X = train_principalDf_xy_1[['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7']] 
Y = train_principalDf_xy_1['Average_Fare']

Xtrain_pc, Xtest_pc, Ytrain_pc, Ytest_pc = train_test_split(X,Y,test_size = 0.30, random_state = 34)

### L1 Regularization

lassoMod = linear_model.Lasso(alpha=0.9) 

lassoMod.fit(Xtrain_pc,Ytrain_pc)

print (lassoMod.coef_)

lassoMod.coef_[((lassoMod.coef_)!=0)]

lassoMod.score(Xtrain_pc,Ytrain_pc)

lassoMod.score(Xtest_pc,Ytest_pc)