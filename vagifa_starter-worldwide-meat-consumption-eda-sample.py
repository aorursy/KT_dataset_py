import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
df = pd.read_csv('/kaggle/input/meat_consumption_worldwide.csv')

df.head(5)
df.info()
df.describe()
df.nunique()
df.skew()
df = pd.get_dummies(df,columns=['MEASURE'])

df
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')

plt.show()
sns.distplot(np.log1p(df['Value']))

plt.show()
by_c = df.groupby('LOCATION')[['Value']].sum().reset_index().sort_values('Value',ascending=False)
px.bar(by_c,by_c['LOCATION'],by_c['Value'])
px.bar(by_c[:10],by_c[:10]['LOCATION'],by_c[:10]['Value'],labels={'x':'Country','y':'Value'})
by_t = df.groupby(['TIME'])[['Value']].sum().reset_index().sort_values('Value',ascending=False)
px.bar(by_t,by_t['TIME'],by_t['Value'])
by_s = df.groupby('SUBJECT')[['Value']].sum().reset_index().sort_values('Value',ascending=False)
px.bar(by_s,by_s['SUBJECT'],by_s['Value'])