#importing libraries

import pandas as pd

import numpy as np

import scipy as sp

from numpy import mean

from numpy import std

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

%matplotlib inline

from sklearn import model_selection

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split as sklearn_train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

import statsmodels.api as sm

#Loading the data

daydata=pd.read_csv('../input/bike-rental-daily/day.csv')

daydata.head()
#As instant just appears to be an index so we can drop the column

daydata.drop('instant',inplace=True,axis=1)
print('The shape of data is-',daydata.shape)
daydata.info()
#Checking for duplicate rows

print('No.of duplicate records=',daydata.duplicated().sum())
#Exploring the distribution of target variable

daydata.cnt.value_counts()/daydata.shape[0]
from scipy.stats import norm

sns.distplot(daydata['cnt'],fit=norm,kde=False)
#Converting the date to datetime

daydata['dteday']=daydata['dteday'].apply(pd.to_datetime)
#Describing the stats

daydata.describe()
cols=daydata.select_dtypes(exclude=['datetime64[ns]']).columns



def dist_plots(size,cols,data):

    plt.figure(figsize=size)

    for i in range(len(cols)):

        plt.subplot(7,2,i+1)

        sns.distplot(data[cols[i]],color='y',fit=norm,kde=False)

        

dist_plots((15,25),cols,daydata)
features=daydata.select_dtypes(exclude=['datetime64[ns]']).columns



def box_plots(size,cols,data):

    plt.figure(figsize=size)

    for i in range(len(features)):

        plt.subplot(7,2,i+1)

        sns.boxplot(x=data[features[i]])

        

        

box_plots((20,25),features,daydata)
cols=['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',

       'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual',

       'registered', 'cnt']



def outlier(col):

    

    #for each in cols:

    stat = daydata[col].describe()

    print(stat)

    IQR = stat['75%'] - stat['25%']

    upper = stat['75%'] + 1.5 * IQR

    lower = stat['25%'] - 1.5 * IQR

    print('The upper and lower bounds for suspected outliers are {} and {}.'.format(upper, lower))
#For season feature

outlier('season')
#For yr feature

outlier('yr')
daydata[(daydata['yr']>2.5)|(daydata['yr']<-1.5)]
#For holiday feature

outlier('holiday')
daydata[(daydata['holiday']>0)|(daydata['holiday']<1)]
#For humidity feature

outlier('hum')
daydata[(daydata['hum']< 0.20 )]
daydata=daydata[(daydata['hum']> 0.20 )]
outlier('windspeed')
#For windspeed feature

daydata=daydata[(daydata['windspeed']<0.44)]
#Cross correlation



def correlation_matrix(data):

    corr=data.corr()

    plt.figure(figsize=(20,10))

    sns.heatmap(corr,annot=True,fmt='.1g',xticklabels=corr.columns.values,yticklabels=corr.columns.values,cmap="YlGnBu",cbar=False)



correlation_matrix(daydata)
daydata.drop(['dteday',"casual","registered"],inplace=True,axis=1)
#Splitting the data

def split(df):

    

    X,y= df.loc[:,df.columns!='cnt'],df['cnt']

    

    X_train,X_test,y_train,y_test=sklearn_train_test_split(X,y,test_size=0.25,random_state=42)

    

    #Scaling the data

    scaler=StandardScaler()

    scaler.fit(X_train)

    split.transformed_X_train=scaler.transform(X_train)

    split.transformed_X_test= scaler.transform(X_test)

    return X_train,X_test,y_train,y_test
X_train,X_test,y_train,y_test=split(daydata)
    

def model(model):

    #Linear regression model

    model= model

    fitted_model=model.fit(X_train,y_train)

    y_pred=fitted_model.predict(X_test)

    r2 = format(r2_score(y_test, y_pred),'.3f')

    rmse = format(np.sqrt(mean_squared_error(y_test, y_pred)),'.3f')

    mae = format(mean_absolute_error(y_test, y_pred),'.3f')



    result=pd.DataFrame({'Model':['Baseline_model'],'R-squared':[r2],'RMSE':[rmse],'MAE':[mae]})

    return result

result=model(LinearRegression())
result
#Checking for multicollinearity

def multicollinearity():

    X= daydata.loc[:,daydata.columns!='cnt']

    X_vif= X

    print(pd.Series([variance_inflation_factor(X_vif.values,i) for i in range (X_vif.shape[1])],index=X_vif.columns))

    split(daydata)
multicollinearity()
#Removing the highest correlated feature and observing.

daydata.drop('atemp',inplace=True,axis=1)


multicollinearity()
#Removing humidity as it is the highest correlated factor.

daydata.drop(['hum'],inplace=True,axis=1)
multicollinearity()
#Removing season

daydata.drop(['season'],inplace=True,axis=1)
multicollinearity()
#Removing weathersit

daydata.drop(['weathersit'],inplace=True,axis=1)
multicollinearity()
#Fitting improvised Linear model



model=LinearRegression()

fitted_model=model.fit(X_train,y_train)

y_pred=fitted_model.predict(X_test)

cv=KFold(n_splits=5,shuffle=True,random_state=1)

score=cross_val_score(model, X_train,y_train, scoring='r2', cv=cv, n_jobs=-1)

scores_2 =cross_val_score(model,X_train,y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('R2 with cross val: %.3f (%.3f)' % (mean(score)*100, std(score)*100))

print('MSE with cross val: %.3f (%.3f)' % (mean(scores_2)*100, std(scores_2)*100))

r2 = format(r2_score(y_test, y_pred),'.3f')

rmse = format(np.sqrt(mean_squared_error(y_test, y_pred)),'.3f')

mae = format(mean_absolute_error(y_test, y_pred),'.3f')

result_2 = pd.DataFrame({'Model':['Improvised_model'],'R-squared':[r2],'RMSE':[rmse],'MAE':[mae]})

result_new = result.append(result_2)

result_new
#Random Forest Regressor

model=RandomForestRegressor(n_estimators = 1000,random_state=1233)

fitted_model=model.fit(X_train,y_train)

y_pred=fitted_model.predict(X_test)

cv=KFold(n_splits=5,shuffle=True,random_state=1)

score=cross_val_score(model, X_train,y_train, scoring='r2', cv=cv, n_jobs=-1)

scores_2 =cross_val_score(model,X_train,y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

print('R2 with cross val: %.3f (%.3f)' % (mean(score)*100, std(score)*100))

print('MSE with cross val: %.3f (%.3f)' % (mean(scores_2)*100, std(scores_2)*100))

r2 = format(r2_score(y_test, y_pred),'.3f')

rmse = format(np.sqrt(mean_squared_error(y_test, y_pred)),'.3f')

mae = format(mean_absolute_error(y_test, y_pred),'.3f')

result_2 = pd.DataFrame({'Model':['Improvised_model'],'R-squared':[r2],'RMSE':[rmse],'MAE':[mae]})

result_new = result.append(result_2)

result_new
#Checking the linear fit

fig,ax=plt.subplots()



ax.scatter(y_test,y_pred)

ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=2)

ax.set_xlabel('Actual')

ax.set_ylabel('Predicted')

plt.show()
def OLS_model(df):

    

    X,y= df.loc[:,df.columns!='cnt'],df['cnt']



    lm=sm.OLS(y,X).fit()

    return lm.summary()
OLS_model(daydata)
#Removing workingday as it maybe acting as noice

daydata.drop('workingday',inplace=True,axis=1)
OLS_model(daydata)