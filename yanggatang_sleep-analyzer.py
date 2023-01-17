# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plot

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sleep = pd.read_csv("../input/applehealth/convertcsv.csv") 
sleep.head()
sleep_raw = sleep[sleep['_sourceName'] == 'Sleep Cycle']

sleep_cycle = sleep_raw[['_sourceName', '_creationDate', '_startDate', '_endDate','_value']]
sleep_cycle.shape
sleep_cycle.head()
def elapsed_time(stime,etime): # in minutes
    t1 = pd.to_datetime(stime)
    t2 = pd.to_datetime(etime)
    return pd.Timedelta(t2 - t1).seconds / 60.0 
    
    # pd.Timedelta(t2 - t1).seconds / 3600.0 hours
def convert_time(time):
    raw = time.split(" ")[1]
    hours,minutes,seconds = raw.split(":")
    convert = {}
    convert['20'] = 8
    convert['21'] = 9
    convert['22'] = 10
    convert['23'] = 11
    convert['00'] = 12
    convert['01'] = 13
    convert['02'] = 14
    convert['03'] = 15
    convert['04'] = 16
    convert['05'] = 17
    convert['06'] = 18
    convert['07'] = 19
    convert['08'] = 20
    result = convert[hours] + int(minutes)/60.0 
    return result
    # if you're sleeping after 4 am.... idk bro, you need help
    
def convert_date(time):
    return time.split(" ")[0]
elapsed_time('2020-08-09 23:38:40 -0700','2020-08-10 07:10:34 -0700')
convert_time('2020-08-10 22:55:35 -0700')
sleep_cycle['minutes_elapsed'] = sleep_cycle.apply(lambda row: elapsed_time(row._startDate,row._endDate), axis=1)
sleep_cycle['hours_elapsed'] = sleep_cycle.apply(lambda row: elapsed_time(row._startDate,row._endDate)/60.0, axis=1)
sleep_cycle['start_bed'] = sleep_cycle.apply(lambda row: convert_time(row._startDate), axis=1)
sleep_cycle['date'] = sleep_cycle.apply(lambda row: convert_date(row._startDate), axis=1)
sleep_cycle.head()
column_1 = sleep_cycle["start_bed"]
column_2 = sleep_cycle["hours_elapsed"]
correlation = column_1.corr(column_2)
print(correlation)
dataFrame = pd.DataFrame(data=sleep_cycle, columns=['start_bed','hours_elapsed']);

# Draw a scatter plot

dataFrame.plot.scatter(x='start_bed', y='hours_elapsed', title= "start time in bed vs minutes elapsed");

plot.show(block=True);
def percent_asleep(_valx, _valy, timex,timey): # return percentage asleep
    if(_valx == 'HKCategoryValueSleepAnalysisAsleep' ):
        return timex/timey
    else:
        return timey/timex
    
sleep_filtered = sleep_cycle[sleep_cycle['_value'] == 'HKCategoryValueSleepAnalysisInBed']
sleep_filtered2 = sleep_cycle[sleep_cycle['_value'] == 'HKCategoryValueSleepAnalysisAsleep']

sleep_final = pd.merge(sleep_filtered, sleep_filtered2, on=['date'])
sleep_final['percent_asleep'] = sleep_final.apply(lambda row: percent_asleep(row._value_x,row._value_y,row.hours_elapsed_x, row.hours_elapsed_y), axis=1)
sleep_final['start_bed'] = sleep_final.apply(lambda row: min(row.start_bed_x, row.start_bed_y), axis=1)
sleep_final = sleep_final[sleep_final['percent_asleep'] < 1]
sleep_final.head()
column_1 = sleep_final["start_bed"]
column_2 = sleep_final["percent_asleep"]
correlation = column_1.corr(column_2)
print(correlation)
sleep_final
dataFrame = pd.DataFrame(data=sleep_final, columns=['start_bed','percent_asleep']);

# Draw a scatter plot

dataFrame.plot.scatter(x='start_bed', y='percent_asleep', title= "start time in bed vs percentage asleep");

plot.show(block=True);
import matplotlib.pyplot as plt
import pandas as pd

df = sleep_final

# gca stands for 'get current axis'
ax = plt.gca()

#df.plot(kind='line',x='date',y='hours_elapsed_x',ax=ax)
#df.plot(kind='line',x='date',y='start_bed_x', color='red', ax=ax)
#df.plot(kind='line',x='date',y='hours_elapsed_y',ax=ax)
#df.plot(kind='line',x='date',y='percent_asleep', color='blue', ax=ax)

plt.figure(figsize=(20,12))

plt.show()
# SES example
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# contrived dataset
data = list(sleep_final["percent_asleep"])

# fit model
model = SimpleExpSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)
import sklearn.metrics as metrics
def regression_results(y_true, y_pred):
# Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit()
X = sleep_final[[ "start_bed_x",'hours_elapsed_y',"hours_elapsed_x"]]
y = sleep_final['percent_asleep']

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('NN', MLPRegressor(solver = 'lbfgs')))  #neural network
models.append(('KNN', KNeighborsRegressor())) 
models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees
models.append(('SVR', SVR(gamma='auto'))) # kernel = linear
# Evaluate each model in turn
results = []
names = []
for name, model in models:
    # TimeSeries Cross validation
    
    cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
model = LinearRegression()
model.fit(X_train,y_train)


predicted_percent = model.predict(X_test)
print(predicted_percent)


# data = [[7.683,10.83,7.38], [7.95,10.72,6.8]] 
# custom_test = pd.DataFrame(data, columns = ['hours_elapsed_x', 'start_bed_x','hours_elapsed_y']) 

# print(model.predict(custom_test))
import matplotlib.pyplot as plt
import pandas as pd

training = X_train.join(y_train)
testing = X_test.join(y_test)
df2 = pd.concat([training,testing])
df2 = df2.join(sleep_final['date'])

df = sleep_final


#gca stands for 'get current axis'
ax = plt.gca()

df.plot(kind='line',x='date',y='percent_asleep', color='blue', ax=ax) #actual
#df2.plot(kind='line',x='date',y='percent_asleep', color='green', ax=ax) #predicted

plt.figure(figsize=(20,12))

plt.show()
