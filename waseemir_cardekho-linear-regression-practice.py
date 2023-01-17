#Import Relevent Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

sns.set()
#Import dataset 

data=pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')

x1,y=data['Kms_Driven'],data['Present_Price']

data.describe()
#Analyze the dataset

plt.scatter(x1,y)

plt.xlabel('Kms driven')

plt.ylabel('Present Price')

plt.show()
#Perform Regression

x=sm.add_constant(x1)

results=sm.OLS(y,x).fit()  #Ordinary Least Squares tries to find the line of best fit

results.summary() #Print summary of your results 