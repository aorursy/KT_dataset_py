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
mergedDateTime = inputData.Date +' '+inputData.Time
inputData.index = (pd.to_datetime(mergedDateTime))
inputData.drop(["Time","Date","Transaction"],axis=1,inplace=True)
inputData.head()

startDate,endDate = "2016-10","2017-02"
trainingData = inputData[startDate:endDate]
# Group the data on hourly spans
group = pd.DataFrame({"ItemCount":trainingData.groupby([trainingData.index.map(lambda t: t.hour),"Item"]).size()}).reset_index();

# Now lets find the probabilities
# Note this is an hourly probability - so we only consider items falling within
# the hour period
itemName = "Coffee"
trainItemProbability = group[group["Item"] == itemName]
trainItemProbability.rename(index=int, columns={"level_0": "Hour"},inplace=True)
trainItemProbability.drop(["Item"],axis= 1,inplace=True)
total = np.float(trainItemProbability["ItemCount"].sum())  
trainItemProbability["ItemCount"]=trainItemProbability["ItemCount"].apply(lambda v:(v /total))

# Since the item span may not be in the 24 hours range,
# we add in averages samples. This will ne done to testing data as well
# Will help us keep sanity during testing

hours = np.arange(0,24,1)
trainItemProbability24hrs = pd.DataFrame(0,index=hours,columns=trainItemProbability.columns.values)
def expand_to_24_hours(data24hrs,data):
    for row in range(0,len(data)):
        oneRow = data.iloc[row,:]
        data24hrs.iloc[np.int(oneRow.Hour)] = oneRow
    return data24hrs
trainItemProbability = expand_to_24_hours(trainItemProbability24hrs,trainItemProbability)
x = trainItemProbability["Hour"]
y = trainItemProbability["ItemCount"]

del total,group


testingData = inputData[endDate:]  
group = pd.DataFrame({"ItemCount":testingData.groupby([testingData.index.map(lambda t: t.hour),"Item"]).size()}).reset_index();

testItemProbability = group[group["Item"] == itemName]
testItemProbability.rename(index=int, columns={"level_0": "Hour"},inplace=True)
testItemProbability.drop(["Item"],axis= 1,inplace=True)
total = np.float(testItemProbability["ItemCount"].sum())  
testItemProbability["ItemCount"]=testItemProbability["ItemCount"].apply(lambda v:(v /total))



hours = np.arange(0,24,1) # This will be our new index
testItemProbability24hrs = pd.DataFrame(0,index=hours,columns=testItemProbability.columns.values)
testItemProbability = expand_to_24_hours(testItemProbability24hrs,testItemProbability)

fig = plt.figure(figsize = (15,5))
ax = fig.gca()
xTrain = trainItemProbability24hrs["Hour"]
yTrain = trainItemProbability24hrs["ItemCount"]
xTest = testItemProbability24hrs["Hour"]
yTest = testItemProbability24hrs["ItemCount"]
plt.scatter(xTrain,yTrain,label="Train")
plt.scatter(xTest,yTest,label="Test")
plt.legend()
plt.xlabel('Time span',fontsize=10)
plt.ylabel('Probability',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Probability of {} sold during the day'.format(itemName),fontsize=20)
plt.grid()
plt.ioff()
plt.show()


# Lets also select particular hour range for our predictions
startHour = 3
endHour = 22
condition = np.logical_and((trainItemProbability24hrs.index >= startHour),(trainItemProbability24hrs.index <= endHour))
trainSamples = trainItemProbability24hrs[condition]
condition = np.logical_and((testItemProbability24hrs.index >= startHour),(testItemProbability24hrs.index <= endHour))
testSamples = testItemProbability24hrs[condition]
# Naive 

predicted = trainSamples.copy();
predicted["Naive"] = trainSamples.ItemCount
fig = plt.figure(figsize = (15,5))
ax = fig.gca()
plt.scatter(trainSamples.index,trainSamples.ItemCount,label="Train",c='r',marker='.')
plt.scatter(testSamples.index,testSamples.ItemCount,label="Test",c='g',marker='+')
plt.scatter(predicted.index,predicted.ItemCount,label="Predicted",c='b',marker='*')
plt.legend()
plt.xlabel('Time span',fontsize=10)
plt.ylabel('Probability',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Naive Forecast',fontsize=20)
plt.grid()
plt.ioff()
plt.show()

rms = sqrt(mean_squared_error(testSamples.ItemCount, predicted.Naive))
print(rms)

# Simple Average

predicted["Average"] = trainSamples['ItemCount'].rolling(1).mean()
fig = plt.figure(figsize = (15,5))
ax = fig.gca()
plt.scatter(trainSamples.index,trainSamples.ItemCount,label="Train",c='r',marker='.')
plt.scatter(testSamples.index,testSamples.ItemCount,label="Test",c='g',marker='+')
plt.scatter(predicted.index,predicted.Average,label="Predicted",c='b',marker='*')
plt.legend()
plt.xlabel('Time span',fontsize=10)
plt.ylabel('Probability',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Simple Average Forecast',fontsize=20)
plt.grid()
plt.ioff()
plt.show()

rms = sqrt(mean_squared_error(testSamples.ItemCount, predicted.Average))
print(rms)


# Moving Average

predicted["MAverage"] = trainSamples['ItemCount'].rolling(4).mean()
# Find nans and replace. Nans appear as we initialized a DataFrame with zeros
predicted.fillna(0,inplace=True)
fig = plt.figure(figsize = (15,5))
ax = fig.gca()
plt.scatter(trainSamples.index,trainSamples.ItemCount,label="Train",c='r',marker='.')
plt.scatter(testSamples.index,testSamples.ItemCount,label="Test",c='g',marker='+')
plt.scatter(predicted.index,predicted.MAverage,label="Predicted",c='b',marker='*')
plt.legend()
plt.xlabel('Time span',fontsize=10)
plt.ylabel('Probability',fontsize=10)
ax.tick_params(labelsize=10)
plt.title('Moving Average Forecast',fontsize=20)
plt.grid()
plt.ioff()
plt.show()

rms = sqrt(mean_squared_error(testSamples.ItemCount, predicted.MAverage))
print(rms)
