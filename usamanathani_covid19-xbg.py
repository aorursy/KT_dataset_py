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



# Any results you write to the current directory are saved as output.



import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import csv

import numpy as np

import operator

import random

import datetime





import sklearn.discriminant_analysis

import sklearn.linear_model as skl_lm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, classification_report, precision_score

from sklearn import preprocessing

from sklearn import neighbors

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from datetime import timedelta

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.metrics import hamming_loss, accuracy_score 

from pandas import DataFrame

from datetime import datetime

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error



%matplotlib inline

import matplotlib.pyplot as plt



import statsmodels.api as sm

import statsmodels.formula.api as smf
PATH ='/kaggle/input/covid19-global-forecasting-week-4'

datatrain = pd.read_csv(f'{PATH}/train.csv')

datatest = pd.read_csv(f'{PATH}/test.csv')
date = pd.to_datetime(datatrain["Date"])

datet = pd.to_datetime(datatest["Date"])

print (date)

ldate = int(len(date))

ldatet = int(len(datet))

print("Length of training- date is", ldate)

print("Length of test- date is", ldatet)
m = []

d = []

for i in range(0,ldate):

    dx = (date[i].strftime("%d"))

    mx = (date[i].strftime("%m"))

    m.append(int(mx))

    d.append(int(dx))



mt = []

dt = []

for i in range(0,ldatet):

    dtx = (datet[i].strftime("%d"))

    mtx = (datet[i].strftime("%m"))

    mt.append(int(mtx))

    dt.append(int(dtx))

train = datatrain

test = datatest

train.insert(6,"Month",m,False)

train.insert(7,"Day",d,False)

test.insert(4,"Month",mt,False)

test.insert(5,"Day",dt,False)
train.head()
test.head()
print("Datatrain")

traindays = datatrain['Date'].nunique()

print("Number of Country_Region: ", datatrain['Country_Region'].nunique())

print("Number of Province_State: ", datatrain['Province_State'].nunique())

print("Number of Days: ", traindays)



notrain = datatrain['Id'].nunique()

print("Number of datapoints in train:", notrain)

lotrain = int(notrain/traindays)

print("L Trains:", lotrain)

print("Datatest")

testdays = datatest['Date'].nunique()

print("Number of Days: ", testdays)

notest = datatest['ForecastId'].nunique()

print("Number of datapoints in test:", notest)

lotest = int(notest/testdays)

print("L Test:", lotest)

zt = datet[0]

daycount = []

for i in range(0,lotrain):

    for j in range(1,traindays+1):

        daycount.append(j)



for i in range(traindays):

    if(zt == date[i]):

        zx = i

        print(zx)

        

daytest = []

for i in range(0,lotest):

    for j in range(1,testdays+1):

        jr = zx + j

        daytest.append(jr)

train.insert(8,"DayCount",daycount,False)

test.insert(6,"DayCount",daytest,False)
train.head()
test.head()
traincount = int(len(train["Date"]))



testcount = int(len(test["Date"]))



train.Province_State = train.Province_State.fillna(0)

empty = 0

for i in range(0,traincount):

    if(train.Province_State[i] == empty):

        train.Province_State[i] = train.Country_Region[i]
test.Province_State = test.Province_State.fillna(0)

empty = 0

for i in range(0,testcount):

    if(test.Province_State[i] == empty):

        test.Province_State[i] = test.Country_Region[i]
label = preprocessing.LabelEncoder()

train.Country_Region = label.fit_transform(train.Country_Region)

train.Province_State = label.fit_transform(train.Province_State)
test.Country_Region = label.fit_transform(test.Country_Region)

test.Province_State = label.fit_transform(test.Province_State)

X = np.c_[train["Province_State"], train["Country_Region"], train["DayCount"], train["Month"], train["Day"]]

Xt = np.c_[test["Province_State"], test["Country_Region"], test["DayCount"], test["Month"], test["Day"]]
Y1 = train["ConfirmedCases"]

Y2 = train["Fatalities"]
#Tuning XGB

nest = [23,24,25]

alpha = [0]

gam = [0] #,1,2,3,4,5,6]

lr = [2,3,4]

dep = [23]#[18,19,20,21,22,23,24,25,26]

mscore = 1





for a in range(0,len(alpha)):

    for b in range(0,len(gam)):

        for c in range(0,len(lr)):

            for d in range(0,len(dep)):

                 for e in range(0,len(nest)):

                    a0 = alpha[a]; g0 = gam[b]; l0 = lr[c]/100; 

                    d0 = dep[d]

                    n0 = nest[e] * 100

                    xgb = XGBRegressor(n_estimators = n0 , alpha = a0, gamma = g0, learning_rate = l0,  random_state = 42 , max_depth = d0)

                    xgb.fit(X,Y1)

                    yscore = xgb.predict(X)

                    yscore = np.round(yscore)

                    ascore = mean_squared_error(yscore,Y1)

                    print("Accuracy", ascore)

                    if(ascore<mscore):

                        mscore = ascore

                        best = [a0,g0,l0,d0,n0]

                        print("New Max Accuracy is ",mscore, "alpha", a0, "for gamma", g0, "learning rate", l0, "depth", d0, "n_estimators", n0);

print("Best Estimators are, alpha:", best[0], "Gamma:", best[1], "learning rate", best[2], "depth", best[3], "n_estimators", best[4] );
regr = XGBRegressor(n_estimators = best[4] , alpha = best[0], gamma = best[1], learning_rate = best[2],  random_state = 42 , max_depth = best[3])

regr1 = XGBRegressor(n_estimators = best[4] , alpha = best[0], gamma = best[1], learning_rate = best[2],  random_state = 42 , max_depth = best[3])
#regr = XGBRegressor(n_estimators = 2300 , alpha = 0, gamma = 0, learning_rate = 0.04,  random_state = 42 , max_depth = 23) #Training (ConfirmedCases) - Mean Squared Error is:  0.0005750290495952613

#regr1 = XGBRegressor(n_estimators = 2300 , alpha = 0, gamma = 0, learning_rate = 0.04,  random_state = 42 , max_depth = 23)



#regr = XGBRegressor(n_estimators = 2300 , alpha = 0, gamma = 0, learning_rate = 0.03,  random_state = 42 , max_depth = 23) # Best

#regr1 = XGBRegressor(n_estimators = 2300 , alpha = 0, gamma = 0, learning_rate = 0.03,  random_state = 42 , max_depth = 23) 
regr.fit(X,Y1.ravel())

yscore = regr.predict(X)

ascore = mean_squared_error(yscore,Y1)

print("Training (ConfirmedCases) - Mean Squared Error is: ",ascore)
#regr = XGBRegressor(n_estimators = 2250 ,min_child_weight = 1, gamma = 1.15, learning_rate = 0.035,  random_state = 42 , max_depth = 23) # 0.0055

#regr1 = XGBRegressor(n_estimators = 2250 ,min_child_weight = 1, gamma = 1.15, learning_rate = 0.035,  random_state = 42 , max_depth = 23) 
ypred = regr.predict(Xt)

ypred = pd.DataFrame({'ConfirmedCases' : ypred}) 

ypred = round(ypred)

ypred.head(20) 
regr1.fit(X,Y2.ravel())

ypred2= regr1.predict(Xt)

yptest = regr1.predict(X)

yptest = np.round(yptest)

ascore = mean_squared_error(yptest,Y2)

print("Training - (Fatalities) Mean Squared Error is", ascore)
ypred2 = pd.DataFrame({'Fatalities' : ypred2}) 

ypred2 = round(ypred2)

ypred2.head(20) # 6 6 7 7 11
ypred2.head(15)
ypc = pd.DataFrame()

forecast = test["ForecastId"]


#yp = yp.drop(['Country_Region', 'Date', 'Month', 'Day', 'DayCount', 'Province_State'], axis=1)

ypc.insert(0,"ForecastId",forecast,False)

ypc.insert(1,"ConfirmedCases",ypred,False)

ypc.insert(2, "Fatalities",ypred2,False)
ypc
print(ypc)
ypc.to_csv('submission.csv', index=False)