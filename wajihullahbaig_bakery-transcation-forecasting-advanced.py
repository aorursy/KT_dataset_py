# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

import os
print(os.listdir("../input"))
inputData=pd.read_csv(r"../input/BreadBasket_DMS.csv")
print(inputData.info())
print(inputData.describe())
print(inputData.head())

mergedDateTime = inputData.Date +' '+inputData.Time
inputData.index = (pd.to_datetime(mergedDateTime))
inputData.drop(["Time","Date","Transaction"],axis=1,inplace=True)

startDate,endDate = "2016-10","2016-12"
trainingData = inputData[startDate:endDate]
testingData = inputData[endDate:]
itemName = "Jam"
#Training Data
dates = trainingData.rename_axis('Dates').index.floor('H')
trainHourlyCount = trainingData.groupby([dates,'Item']).size().reset_index(name='count')
print (trainHourlyCount)

dates = trainingData.rename_axis('Dates').index.floor('24H')
train24HourlyCount = trainingData.groupby([dates,'Item']).size().reset_index(name='count')
print (train24HourlyCount)

# Calculate the probability
total = train24HourlyCount[train24HourlyCount.Item == itemName]["count"].sum()
trainProbability = trainHourlyCount[trainHourlyCount.Item == itemName].copy()
trainProbability["HourlyProbability"] = trainHourlyCount[trainHourlyCount.Item == itemName]["count"]/total

#Testing Data
dates = testingData.rename_axis('Dates').index.floor('H')
testHourlyCount = testingData.groupby([dates,'Item']).size().reset_index(name='count')
print (testHourlyCount)

dates = testingData.rename_axis('Dates').index.floor('24H')
test24HourlyCount = testingData.groupby([dates,'Item']).size().reset_index(name='count')
print (train24HourlyCount)

total = test24HourlyCount[test24HourlyCount.Item == itemName]["count"].sum()
testProbability = testHourlyCount[testHourlyCount.Item == itemName].copy()
testProbability["HourlyProbability"] = testHourlyCount[testHourlyCount.Item == itemName]["count"]/total
fig = plt.figure(figsize = (15,5))
ax = fig.gca()
x = list(trainProbability.Dates.dt.time)
y = trainProbability["HourlyProbability"]
plt.scatter(x,y,label="Train")

x = list(testProbability.Dates.dt.time)
y = testProbability["HourlyProbability"]
plt.scatter(x,y,label="Test")

plt.legend()
plt.xlabel('Time span',fontsize=10)
plt.ylabel('Probability',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Probability of {} sold during the day'.format(itemName),fontsize=20)
plt.grid()
plt.ioff()
plt.show()
import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing

# We need datetime index for the analysis
trainProbability.set_index("Dates",inplace=True)
trainProbability.drop(["Item","count"],axis=1,inplace=True)

testProbability.set_index("Dates",inplace=True)
testProbability.drop(["Item","count"],axis=1,inplace=True)
sm.tsa.seasonal_decompose(trainProbability.HourlyProbability,freq=1).plot()
result = sm.tsa.stattools.adfuller(trainProbability.HourlyProbability)
plt.show()
predicted = testProbability.copy()
fit1 = Holt(np.asarray(trainProbability.HourlyProbability)).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
predicted['Holt_linear'] = fit1.forecast(len(testProbability))
fig = plt.figure(figsize = (15,5))
ax = fig.gca()
plt.scatter(pd.Series(trainProbability.index.time).astype(str),trainProbability.HourlyProbability,label="Train",c='r',marker='.')
plt.scatter(pd.Series(testProbability.index.time).astype(str),testProbability.HourlyProbability,label="Test",c='g',marker='+')
plt.scatter(pd.Series(predicted.index.time).astype(str),predicted.Holt_linear,label="Predicted",c='b',marker='*')
plt.legend()
plt.xlabel('Time span',fontsize=10)
plt.ylabel('Probability',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Holt\'s Linear Forecast',fontsize=20)
plt.grid()
plt.ioff()
plt.show()

rms = sqrt(mean_squared_error(testProbability.HourlyProbability, predicted.Holt_linear))
print("RMS Error:",rms)

# Holt-Winters linear trend method
del predicted
predicted = testProbability.copy()
fit1 = ExponentialSmoothing(np.asarray(trainProbability.HourlyProbability) ,seasonal_periods=6,trend='add', seasonal='add',).fit()
predicted['Holt_Winters'] = fit1.forecast(len(testProbability))
fig = plt.figure(figsize = (15,5))
ax = fig.gca()
plt.scatter(pd.Series(trainProbability.index.time).astype(str),trainProbability.HourlyProbability,label="Train",c='r',marker='.')
plt.scatter(pd.Series(testProbability.index.time).astype(str),testProbability.HourlyProbability,label="Test",c='g',marker='+')
plt.scatter(pd.Series(predicted.index.time).astype(str),predicted.Holt_Winters,label="Predicted",c='b',marker='*')
plt.legend()
plt.xlabel('Time span',fontsize=10)
plt.ylabel('Probability',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Holt-Winters\'s Forecast',fontsize=20)
plt.grid()
plt.ioff()
plt.show()

rms = sqrt(mean_squared_error(testProbability.HourlyProbability, predicted.Holt_Winters))
print("RMS Error:",rms)
