import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

sns.set()
data=pd.read_csv('../input/real_estate_price_size_year.csv')

data.head(3)
data.describe()
y=data['price']

x1=data[['size','year']]
x=sm.add_constant(x1)

results=sm.OLS(y,x).fit()

results.summary()