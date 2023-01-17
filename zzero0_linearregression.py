import pandas as pd
df=pd.read_csv("../input/CryptocoinsHistoricalPrices.csv")
df.head()
df.drop("Unnamed: 0",axis=1,inplace=True) #removing unnecessary column
df.info()
df["Date"] = pd.to_datetime(df["Date"])  #'Date' column was in plain string form
df["Market.Cap.New"] = [i.replace(',','') for i in df["Market.Cap"]]
df["Volume.New"] = [i.replace(',','') for i in df["Volume"]]
#found some garbage elements needed to be removed

df = df[df["Market.Cap"]!='-']

df = df[df["Volume"]!='-']
df = df[df["Market.Cap"]!='No data was found for the selected time period.']
df["Market.Cap.New"] = df["Market.Cap.New"].astype('int') #conversion to int
df["Volume.New"] = df["Volume.New"].astype('int')
df.info()
df.head()
df_btc = df[df['coin']=='BTC']
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.pairplot(data=df_btc)
df_btc_feature = df_btc[["Open","High","Low","Close","Volume.New","Delta"]]
df_btc_target = df_btc["Market.Cap.New"]
plt.hist(df_btc_target,bins=15)
import numpy as np



df_btc_target_norm = np.log1p(df_btc_target) #log_transfermation
plt.hist(df_btc_target_norm,bins=15)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
x1_train,x1_test,y1_train,y1_test = train_test_split(df_btc_feature,df_btc_target,test_size=0.33,random_state=58) #no log transformation
x_train,x_test,y_train,y_test = train_test_split(df_btc_feature,df_btc_target_norm,test_size=0.33,random_state=58) #log trnaformation
linear1 = LinearRegression(normalize=True).fit(x1_train,y1_train)
linear = LinearRegression().fit(x_train,y_train)
linear1.score(x1_train,y1_train)
linear1.score(x1_test,y1_test)
linear.score(x_train,y_train)
linear.score(x_test,y_test)
predict_y1 = linear1.predict(x1_test)
predict_y= linear.predict(x_test)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,predict_y)
mean_squared_error(y1_test,predict_y1)


plt.scatter(predict_y,predict_y-y_test)

plt.hlines(y=0,xmin=0,xmax=50)
plt.scatter(predict_y1,predict_y1-y1_test)

plt.hlines(y=0,xmin=0,xmax=50)
plt.scatter(x1_test['Open'],y1_test, color='black')

plt.plot(x1_test['Open'],predict_y1,color='blue', linewidth=0.3)
plt.scatter(x_test['Open'],y_test, color='black')

plt.plot(x_test['Open'],predict_y,color='blue', linewidth=0.3)

'''plt.scatter(x_test['Open'],y_test, color='black')

plt.plot(np.log1p(x_test['Open']),np.log1p(predict_y),color='blue', linewidth=0.3)

'''
plt.hist((predict_y1-y1_test)**2)
plt.hist((predict_y-y_test)**2)
plt.hist(predict_y1-y1_test)
plt.hist(predict_y-y_test)
from sklearn.linear_model import Ridge
xR1_train,xR1_test,yR1_train,yR1_test = train_test_split(df_btc_feature,df_btc_target,test_size=0.33,random_state=99)
xR_train,xR_test,yR_train,yR_test = train_test_split(df_btc_feature,df_btc_target_norm,test_size=0.33,random_state=99)
ridge_btc1 = Ridge().fit(xR1_train,yR1_train)
ridge_btc = Ridge().fit(xR_train,yR_train)
ridge_btc1.score(xR1_train,yR1_train)
ridge_btc1.score(xR1_test,yR1_test)
ridge_btc.score(xR_train,yR_train)
ridge_btc.score(xR_test,yR_test)
predict_yR1 = ridge_btc1.predict(xR1_test)
predict_yR = ridge_btc.predict(xR_test)
plt.scatter(predict_yR1,predict_yR1-yR1_test)

plt.hlines(y=0,xmin=0,xmax=50)
plt.scatter(predict_yR,predict_yR-yR_test)

plt.hlines(y=0,xmin=0,xmax=50)
plt.scatter(xR1_test['Open'],yR1_test, color='black')

plt.plot(xR1_test['Open'],predict_yR1,color='blue', linewidth=0.3)
plt.scatter(xR_test['Open'],yR_test, color='black')

plt.scatter(xR_test['Open'],predict_yR,color='blue', linewidth=0.3)
plt.hist((predict_yR1-yR1_test)**2)
plt.hist((predict_yR-yR_test)**2)
plt.hist(predict_yR1-yR1_test)
plt.hist(predict_yR-yR_test)
from sklearn.linear_model import Lasso
xL1_train,xL1_test,yL1_train,yL1_test = train_test_split(df_btc_feature,df_btc_target,test_size=0.33,random_state=99)
xL_train,xL_test,yL_train,yL_test = train_test_split(df_btc_feature,df_btc_target_norm,test_size=0.33,random_state=99)
lasso1 = Lasso().fit(xL1_train,yL1_train)
lasso = Lasso(alpha=0.00000001).fit(xL_train,yL_train)
lasso1.score(xL1_train,yL1_train)
lasso1.score(xL1_test,yL1_test)
lasso.score(xL_train,yL_train)
lasso.score(xL_test,yL_test)
predict_yL1 = lasso1.predict(xL1_test)
predict_yL = lasso.predict(xL_test)
plt.scatter(predict_yL1,predict_yL1-yL1_test)

plt.hlines(y=0,xmin=0,xmax=50)
plt.scatter(predict_yL,predict_yL-yL_test)

plt.hlines(y=0,xmin=0,xmax=50)
plt.scatter(xL1_test['Open'],yL1_test, color='black')

plt.plot(xL1_test['Open'],predict_yL1,color='blue', linewidth=0.3)
plt.scatter(xL_test['Open'],yL_test, color='black')

plt.scatter(xL_test['Open'],predict_yL,color='blue', linewidth=0.3)
predict_yL_norm = lasso.predict(np.log1p(xL_test))
#plt.scatter(np.log1p(xL_test['Open']),yL_test, color='black')

plt.scatter(xL_test['Open'],predict_yL_norm,color='blue', linewidth=0.3)
plt.hist((predict_yL1-yL1_test)**2)
plt.hist((predict_yL-yL_test)**2)
plt.hist((predict_yL1-yL1_test))
plt.hist((predict_yL-yL_test))