import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd 

from matplotlib import rcParams

import plotly.express as px



%matplotlib inline

# figure size in inches

rcParams['figure.figsize'] = 15,6



 



df = pd.read_csv("/kaggle/input/my-dataset/Calcc_COVID-19_train.csv")

do = pd.read_csv("/kaggle/input/my-dataset/Calcc_Crude_oil_trend_From1986-10-16_To2020-03-31.csv")

dff= pd.merge(df, do, on='Date', how='inner')



#print(dff.dtypes)

dff['Date']=pd.to_datetime(dff['Date'])





mask = (dff['Date'] >= '1/1/2020') & (dff['Date'] <= '5/22/2020')

dff=dff.loc[mask]

    

chart1=plt.plot( 'Date', 'World_total_cases_Difference', data=dff, marker='', color='olive', linewidth=2)



fig = px.line(dff, x='Date', y='Price_Difference_Daywise_x')

fig.update_layout(title_text="Daywise Price Drop Trend & Total COVID19 affected increased")

fig.show()