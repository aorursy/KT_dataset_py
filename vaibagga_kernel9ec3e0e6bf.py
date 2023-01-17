import pandas as pd
import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
import itertools
from scipy import stats
%matplotlib inline
## change directory to the directory in which data is present
os.chdir(os.getcwd() + '\\Data')
## reading file
train = pd.read_csv('train.csv')
train.head()
print("Number of data points =", train.shape[0], "\nNumber of Variables =", train.shape[1])
## drop id as it is not a feature
train.drop(['Id'], axis = 1, inplace = True)

## describe method gives the description of each of the variables in the dataframe
train.describe()
## variables to be analyzed
variables = ['O3', 'PM10', 'PM25', 'NO2', 'T2M','mortality_rate']

## colors for different histograms
colors = itertools.cycle(["r", "b", "g", "y", "m", "k"])

## creating plot
f = plt.figure()

f, axes = plt.subplots(nrows = 6, figsize = (30, 60))

## creating subplots
for i in range(len(variables)):
    dist = train[variables[i]].fillna(train[variables[i]].mean())
    sc = axes[i].hist(dist, bins = 20, color = next(colors))
    axes[i].set_xlabel(variables[i])
train.corr()
## heatmap of correlation
sns.heatmap(train.corr())
## features in the dataset
features = ['O3', 'PM10', 'PM25', 'NO2', 'T2M']

## colors for plotting
colors = itertools.cycle(["r", "b", "g", "y", "m"])

## creating plot
f = plt.figure()    

## creating subplots
f, axes = plt.subplots(nrows = 5, figsize = (30, 60))

## Mortality Rate versus O3
sc = axes[0].scatter(train[features[0]], train.mortality_rate, marker = '.', color = next(colors))
axes[0].set_xlabel(features[0], labelpad = 5)

## Mortality Rate versus PM10
sc = axes[1].scatter(train[features[1]], train.mortality_rate, marker = '.', color = next(colors))
axes[1].set_xlabel(features[1], labelpad = 5)

## Mortality Rate versus PM2.5
sc = axes[2].scatter(train[features[2]], train.mortality_rate, marker = '.', color = next(colors))
axes[2].set_xlabel(features[2], labelpad = 5)

## Mortality Rate versus NO2
sc = axes[3].scatter(train[features[3]], train.mortality_rate, marker = '.', color = next(colors))
axes[3].set_xlabel(features[3], labelpad = 5)

## Mortality Rate versus mean temperature
sc = axes[4].scatter(train[features[4]], train.mortality_rate, marker = '.', color = next(colors))
axes[4].set_xlabel(features[4], labelpad = 5)
maharashtra = pd.read_csv('maharashtra.csv')
maharashtra.head()
so2 = (maharashtra.groupby('City/Town/Village/Area')['SO2'].mean())
no2 = (maharashtra.groupby('City/Town/Village/Area')['NO2'].mean())
pm10 = (maharashtra.groupby('City/Town/Village/Area')['RSPM/PM10'].mean())
pm25 = (maharashtra.groupby('City/Town/Village/Area')['PM 2.5'].mean())
'''
Input : 2 distributions (2 pandas series to be compared)
Output : Z-Value (single float)
'''
def ZTest(distribution1 ,distribution2):
    return (np.mean(distribution1) - np.mean(distribution2)) / np.sqrt(np.std(distribution1) ** 2  /  len(distribution1) + np.std(distribution2) ** 2 / len(distribution2))
print(ZTest(maharashtra['NO2'], train['NO2']))
plt.hist(maharashtra['NO2'].fillna(maharashtra['NO2'].mean()), bins = 20, label = 'Maharashtra in 2015')
plt.hist(train['NO2'].fillna(train['NO2'].mean()), bins = 20, label = 'England in 2007-2010')
plt.title('NO2 levels in Maharashtra and England')
plt.legend()
print(ZTest(maharashtra['RSPM/PM10'], train['PM10']))
plt.hist(maharashtra['RSPM/PM10'].fillna(maharashtra['RSPM/PM10'].mean()), bins = 20, label = 'Maharashtra in 2015')
plt.hist(train['PM10'].fillna(train['PM10'].mean()), bins = 20, label = 'England in 2007-2010')
plt.title('PM10 levels in Maharashtra and England')
plt.legend()
## SO2 level
print('List of cities by SO2 level:\n', so2.sort_values()[::-1])
print('List of cities by NO2 level:\n', no2.sort_values()[::-1])
print('List of cities by PM10 level:\n', pm10.sort_values()[::-1])