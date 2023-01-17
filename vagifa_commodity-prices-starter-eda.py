import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt



import numpy as np 

import pandas as pd 
df = pd.read_csv('/kaggle/input/usa-commodity-prices/commodity-prices-2016.csv')

df.head(5)
df.info()
df.isnull().sum()
df.corr()
df.describe()
plt.figure(figsize=(10,10))

sns.heatmap(df.isnull(),yticklabels=False,cmap='rainbow')

plt.show()
px.line(df,df['Date'],df['Zinc'])
px.line(df,df['Date'],df['Tin'])
px.line(df,df['Date'],df['Tea'])
px.histogram(df['Tea'])
px.histogram(df['Tin'])
px.histogram(df['Zinc'])
px.line(df,df['Date'],df['Rice'])
px.histogram(df['Rice'])
px.line(df,df['Date'],df['Natural Gas - Russian Natural Gas border price in Germany'])
px.line(df,df['Date'],df['Natural Gas - Indonesian Liquefied Natural Gas in Japan'])