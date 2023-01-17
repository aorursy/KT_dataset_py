import pandas as pd

import numpy as np

import os
house = pd.read_csv("../input/california-housing-prices/housing.csv")
house.head()
house.info()
house.describe()
house.isnull().sum()
house[house.isnull().any(1)]
house.dropna(inplace=True)
house.isnull().sum()
house[house.duplicated()==True]
from sklearn.model_selection import train_test_split 
import numpy as np

import matplotlib.pyplot as plt
x = house[['latitude','longitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']]
y = house['median_house_value']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=22)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(x_train,y_train)
rfr.score(x_test,y_test)