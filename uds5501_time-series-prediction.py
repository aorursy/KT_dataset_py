# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
init_notebook_mode(connected=True)

import os
print(os.listdir("../input"))
sns.set_style('darkgrid')
# Any results you write to the current directory are saved as output.
os.listdir("../input/csvs_per_year/csvs_per_year/")
myDf = pd.read_csv('../input/csvs_per_year/csvs_per_year/madrid_2016.csv')

print ("IDs of Madrid air monitoring stations")
print(*myDf['station'].unique())
# Fixing datatypes
station1 = myDf[myDf['station'] == 28079008].dropna()
station1 = station1[['date', 'PM10']]
station1['date'] = pd.to_datetime(station1['date'])
station1.sort_values(['date'], inplace = True)
flag = False
for i in range (1, len(station1['date'])):
    if station1['date'].iloc[i-1]  >  station1['date'].iloc[i]:
        print (" Erranious at {}".format(i))
        flag = True
if flag == False:
    print("All Good")

datetime_trace = go.Scatter(
    x = station1['date'],
    y = station1['PM10']
)

data = [datetime_trace]
iplot(data)

from keras.models import Sequential
from keras.layers import Dense
station1.columns
myDataset = pd.DataFrame(station1['PM10'])
Dataset = myDataset.values
Dataset = Dataset.astype('float32')
train_size = int(len(Dataset) * 0.67)
test_size = len(Dataset) - train_size
train, test = Dataset[0:train_size,:], Dataset[train_size:len(Dataset),:]
print(len(train), len(test))
def createDataset(data, look_back = 12):
    dataX, dataY = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
look_back = 12
trainX, trainY = createDataset(train, look_back)
testX, testY = createDataset(test, look_back)
#trainY
# Making a multilayer perceptron
model = Sequential()
model.add(Dense(8, input_dim = look_back ,activation='relu'))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer= 'adam')
model.summary()

model.fit(trainX, trainY, epochs = 100, batch_size = 2, verbose = 2)
import math
trainScore = model.evaluate(trainX, trainY, verbose = 0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
trainPred = model.predict(trainX)
testPred = model.predict(testX)
#test
testPred.flatten()
visGraph = pd.DataFrame({'True': testY, 'Predicted':testPred.flatten()})
Actual_level = go.Scatter(
    x = visGraph.index,
    y = visGraph['True']
)
Predicted_level = go.Scatter(
    x = visGraph.index,
    y = visGraph['Predicted']
)

data = [Actual_level, Predicted_level]
iplot(data)
def myAccuracyTracker(predictions, truth, acceptance = 5):
    totals = len(predictions)
    wrongs = 0
    for i in range(totals):
        if abs(truth[i] - predictions[i] > acceptance):
            wrongs += 1
    return (totals-wrongs)/totals
accuracy_array = []
for acceptance in range(0, 21):
    accuracy_array.append(myAccuracyTracker(testPred.flatten(), testY, acceptance) * 100)
accuracy_array = np.array(accuracy_array)


fig2 = plt.figure(figsize=(15,5))
plt.plot(accuracy_array)
plt.xlabel('Acceptance  Value')
plt.ylabel('Accuracy')

trace_new = go.Scatter(
        y = accuracy_array,
        x = np.array(range(21)),
        name = "Accuracy Curve"
)
layout = dict(title = 'Accuracy and Acceptance Payoff',
              xaxis = dict(title = 'Accuracy (in percentage)'),
              yaxis = dict(title = 'Acceptance (in PPM)'),
              )
mydata = [trace_new]
meow = dict(data = mydata, layout= layout)
iplot(meow)

