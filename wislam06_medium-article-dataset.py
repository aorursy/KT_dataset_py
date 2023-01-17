# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

data = pd.read_csv("../input/medium-articles-dataset/medium_data.csv")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data.head()
#Lets see what we have in our data. 

data.info()
#checks that publication date was a weekend or not
#Also, checks for the what day of the week it was
#This informtaion is not contributing much to our model so we skip ot out.... 

#data['IsWeekend'] = ((pd.DatetimeIndex(data.index).dayofweek) // 5 == 1).astype(int)
#data['weekOfDay'] = ((pd.DatetimeIndex(data.index).dayofweek)).astype(int)

data["HasImage"] = data["image"].apply(lambda x: 0 if x is np.nan else 1)
data["HasSubtitle"] = data["subtitle"].apply(lambda x: 0 if x is np.nan else 1)
data["subtitle"] = data["subtitle"].apply(lambda x: "" if x is np.nan else x)
data["LengthOfTitle"] = data["title"].apply(lambda x: len(x))
data["LengthOfSubtitle"] = data["subtitle"].apply(lambda x: len(x))

data["responses"].replace("Read",0, inplace=True)
data["responses"] = data["responses"].apply(lambda x: int(x))

data = pd.get_dummies(data, columns=['publication'])
data
data.drop(["image","title","subtitle","date","id","url"], axis=1, inplace=True)
data
data.columns

columns = ['claps', 'responses', 'reading_time',
       'LengthOfTitle', 'LengthOfSubtitle']
data.shape

# Plotting data against the claps to check which factors are contributing towards it  

fig = plt.figure(figsize=(12,6))

for i in range(len(columns)):
    fig.add_subplot(2,4,i+1)
    sns.scatterplot(y=data["claps"], x=data[columns[i]])
    
plt.tight_layout()
# Verfying the normality of data

fig = plt.figure(figsize=(12,6))

for i in range(len(columns)):
    fig.add_subplot(2,4,i+1)
    sns.distplot(data[columns[i]], kde_kws={"bw":0.01},  fit=norm, kde=False)
    
plt.tight_layout()
# Checking for outliers 

fig = plt.figure(figsize=(12,6))

for i in range(len(columns)):
    fig.add_subplot(2,4,i+1)
    sns.boxplot(y=data[columns[i]])
    
plt.tight_layout()
print(data)
#Trying Log Trasanformation on the data to make it Normally distributed  

data['claps'] = np.log(data['claps'] + 1)
data['responses'] = np.log(data['responses'] + 1)
data['LengthOfTitle'] = np.log(data['LengthOfTitle'] + 1)
data['reading_time'] = np.log(data['reading_time'] +1 )
data['LengthOfSubtitle'] = np.log(data['LengthOfSubtitle'] + 1)

fig = plt.figure(figsize=(12,6))

for i in range(len(columns)):
    fig.add_subplot(2,4,i+1)
    sns.distplot(data[columns[i]], kde_kws={"bw":0.01},  fit=norm, kde=False)
    
plt.tight_layout()

print(data)
# Detecting outliers using z-score....


z_scores = stats.zscore(data)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data_ZScore = data[filtered_entries]
data_ZScore.shape
print(data_ZScore)
fig = plt.figure(figsize=(12,6))

for i in range(len(columns)):
    fig.add_subplot(2,4,i+1)
    sns.distplot(data_ZScore[columns[i]], kde_kws={"bw":0.01})
    
plt.tight_layout()
#checking after outlier removal 

fig = plt.figure(figsize=(12,6))

for i in range(len(columns)):
    fig.add_subplot(2,4,i+1)
    sns.scatterplot(y=data_ZScore["claps"], x=data_ZScore[columns[i]])
    
plt.tight_layout()
fig = plt.figure(figsize=(12,6))

for i in range(len(columns)):
    fig.add_subplot(2,4,i+1)
    sns.boxplot(y=data_ZScore[columns[i]])
    
plt.tight_layout()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#data = data_IQR    #0.2642893199990747
#data = data_ZScore #0.44418822074518904
data = data        #0.6902850689483697
data.shape
LinerRegressionModel = LinearRegression()
X = data.drop(["claps"], axis=1)
y = data.claps



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 0)


LinerRegressionModel.fit(X_train,y_train)
y_pred = LinerRegressionModel.predict(X_test)

print("R Squared Value:  " ,r2_score(y_test,y_pred))

print("Mean Squared Error:  " ,mean_squared_error(y_test, y_pred))
print("Mean Absolute Error: " ,mean_absolute_error(y_test,y_pred))



#R Squared Value:   0.6177129222296169
#Mean Squared Error:   48.69482050883009
#Mean Absolute Error:  4.896506764886545
sns.regplot(x=y_test, y=y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")

randomForestModel = RandomForestRegressor()

randomForestModel.fit(X_train,y_train)
y_pred = randomForestModel.predict(X_test)

print("R Squared Value:  " ,r2_score(y_test,y_pred))

print("Mean Squared Error:  " ,mean_squared_error(y_test, y_pred))
print("Mean Absolute Error: " ,mean_absolute_error(y_test,y_pred))

#1526950.292894805




