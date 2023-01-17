import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import rcParams
%matplotlib inline
# figure size in inches
#rcParams['figure.figsize'] = 90,15

df = pd.read_csv("/kaggle/input/ntt-data-global-ai-challenge-06-2020/COVID-19_train.csv")
do = pd.read_csv("/kaggle/input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend_From1986-10-16_To2020-03-31.csv")
dff= pd.merge(df, do, on='Date', how='inner')


#for col in dff.columns:
#    print(col)
#print(dff[['Date','World_total_deaths','Price_x']])


g = sns.jointplot(x=dff["China_new_cases"], y=dff["Price_x"], kind='hex', marginal_kws=dict(bins=30, rug=True))
sns.jointplot(x="World_new_deaths", y="Price_x", data=dff);
sns.jointplot(x="World_new_deaths", y="Price_x", data=dff, kind='kde')
sns.lmplot(x='World_total_cases', y='World_total_deaths',data=df,fit_reg=True)


chart1=sns.relplot(x="Date", y="World_total_cases", data=dff)
chart1.fig.set_figwidth(30.27)
chart1.fig.set_figheight(20.7)
chart1.set_xticklabels(rotation=90)
chart1.fig.suptitle('Date wise Overall Death Trend')