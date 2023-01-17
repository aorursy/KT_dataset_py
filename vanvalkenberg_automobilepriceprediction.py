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
####################################################################################################################
#                                             Contents                                                             #
#                                                                                                                  #
#    1.) Clean data and mapping values to generate machine understanable data                                      #
#    2.) Training and comparing various models                                                                     #
#    3.) Comparing performance of various models                                                                   #
#    4.) Visulizations                                                                                             #
####################################################################################################################
DataFrame = pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv')
DataFrame.info()
## lets clean the data 
## and convert objects
## to int 

DataFrame = DataFrame.replace(to_replace = '?', value = 0) 
DataFrame['price'] = DataFrame['price'].astype('int64')
## let define a mapper function
## this function will map objects 
## integer values
def mapper(param):
    uniqueVals = set(DataFrame[param]);
    mappedVals = {};
    i = 0
    
    for mem in uniqueVals:
        mappedVals[mem] = i
        i = i + 1
    
    print (mappedVals)
    DataFrame[param] =  DataFrame[param].map(mappedVals)
    
    return uniqueVals, mappedVals
    
## using above functions lets get done with mapping
uFuelsystem, mFuelsystem = mapper('fuel-system')
## mapping engine-location
uEngineLocation, mEngineLocation = mapper('engine-location')
## mapping drive-wheels
uDriveWheel,  mWheelDrive = mapper('drive-wheels')
## mapping body-style, num-of-doors, aspiration, fuel-type, make
uBodyStyle, mBodyStyle = mapper('body-style')
uNumDoors, mNumDoors = mapper('num-of-doors')
uAspiration, mAspiration = mapper('aspiration')
uFuelSystem, mFuelSystem = mapper('fuel-type')
uMAke, mMake = mapper('make')



## also mapping these features 
uNumOfCylinders, mNumOfCylinders = mapper('num-of-cylinders')
uEngineType, MEngineType = mapper('engine-type')
## remaining work is to  typeCast objects to int 64
DataFrame['normalized-losses'] = DataFrame['normalized-losses'].astype('int64')
DataFrame['engine-type'] = DataFrame['engine-type'].astype('int64')
DataFrame['num-of-cylinders'] = DataFrame['num-of-cylinders'].astype('int64')
DataFrame['stroke'] = DataFrame['stroke'].astype('float64')
DataFrame['bore'] = DataFrame['bore'].astype('float')
DataFrame['horsepower'] = DataFrame['horsepower'].astype('int64')
DataFrame['peak-rpm'] = DataFrame['peak-rpm'].astype('int64')

DataFrame.info()
## saving this file so that it can be further used
DataFrame.to_csv('processed_car_data.csv')
## lets train some model using entire data ie not splitting data into training and testing 
FeatureVector = DataFrame [['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
       'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
       'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
       'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
       'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
       'highway-mpg']]

Price = DataFrame['price']

FeatureVector = np.array(FeatureVector)
Price = np.array(Price)
samples, features = FeatureVector.shape
## lets use standard scaler to scale our feature vector
## also lets reduce price by a scale of 1000

import sklearn
from sklearn.preprocessing import StandardScaler
StdSc = StandardScaler()
FeatureVector = StdSc.fit_transform(FeatureVector)
Price = Price / 1000
## lets traing the simplest model ie linear regression
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(FeatureVector, Price)
LinerRegressionsPredictions = LR.predict(FeatureVector)
LinerRegressionsPredictions = LinerRegressionsPredictions.flatten()
## lets see who did linear regression performed
import matplotlib.pyplot as plt
plt.scatter(Price, LinerRegressionsPredictions)
plt.xlabel('true values scaled down by a factor 1000')
plt.ylabel('predicted values scaled down by a factor 1000')
plt.title('LinearRegression')
plt.plot(np.arange(0,50), np.arange(0, 50), color = 'green')
plt.grid(True)
## lets move  a step further to decission Tree regressor
from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor()
DTR.fit(FeatureVector, Price)
DecissionTreeRegressorPrediction = (DTR.predict(FeatureVector))

plt.scatter(Price, DecissionTreeRegressorPrediction)
plt.xlabel('true values scaled down by a factor 1000')
plt.ylabel('predicted values scaled down by a factor 1000')
plt.title('DecisionTreeRegressor')
plt.plot(np.arange(0,50), np.arange(0, 50), color = 'green')
plt.grid(True)

## Seems nice But did it overFitted our data ??? probably yes
## let try a random forest regressor 
## lets hope to see some less overfitting
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor()
RFR.fit(FeatureVector, Price)
RandomForestsPredictions = RFR.predict(FeatureVector)

plt.scatter(Price, RandomForestsPredictions)
plt.xlabel('true values scaled down by a factor 1000')
plt.ylabel('predicted values scaled down by a factor 1000')
plt.title('Random Forest Regressor')
plt.plot(np.arange(0,50), np.arange(0, 50), color = 'green')
plt.grid(True)

## reduced overfitting
## lets try gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor()
GBR.fit(FeatureVector, Price)
GradientBoostsPrediction = GBR.predict(FeatureVector)

plt.scatter(Price, GradientBoostsPrediction)
plt.xlabel('true values scaled down by a factor 1000')
plt.ylabel('predicted values scaled down by a factor 1000')
plt.title('Gradient Boosting Regressor')
plt.plot(np.arange(0,50), np.arange(0, 50), color = 'green')
plt.grid(True)

