import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv(r'/kaggle/input/nigerian-market-fires-2020/Market Fires 2020.csv')

data.head()
data.dtypes
#lets convert the Date of Fire to Datetime.

data['Date of Fire'] = pd.to_datetime(data['Date of Fire'])
# Lets check the value count for unique state

pd.value_counts(data['State']).plot.bar()
pd.value_counts(data['Reported Causes']).plot.bar()
pd.value_counts(data['Fire put out by']).plot.bar()
pd.value_counts(data['Type of Market']).plot.bar()
data['Month'] = data['Date of Fire'].dt.month

data.head()
pd.value_counts(data['Month']).plot.pie()
#Please comment, Thank you.

#Thank you for the dataset