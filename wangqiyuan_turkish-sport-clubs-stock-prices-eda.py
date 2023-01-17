# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np 

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

from datetime import datetime

#from __future__ import division
FB=pd.read_csv('../input/FENER_MG.csv',names=["Stock","DateTime","Open","High","Low","Close","Volume","0"])

GS=pd.read_csv('../input/GSRAY_MG.csv',names=["Stock","DateTime","Open","High","Low","Close","Volume","0"])

BJK=pd.read_csv('../input/BJKAS_MG.csv',names=["Stock","DateTime","Open","High","Low","Close","Volume","0"])
GS=GS.drop("0",axis=1)

BJK=BJK.drop("0",axis=1)

FB=FB.drop("0",axis=1)
FB['DateTime']=pd.to_datetime(FB['DateTime'])

GS['DateTime']=pd.to_datetime(GS['DateTime'])

BJK['DateTime']=pd.to_datetime(BJK['DateTime'])



GS['Stock']=GS['Stock'].astype(str).str[6:]

FB['Stock']=FB['Stock'].astype(str).str[6:]

BJK['Stock']=BJK['Stock'].astype(str).str[6:]

FB.index=FB.DateTime

GS.index=GS.DateTime

BJK.index=BJK.DateTime



FB=FB.drop("DateTime",axis=1)

GS=GS.drop("DateTime",axis=1)

BJK=BJK.drop("DateTime",axis=1)
BJK.head()
FB.head()
GS.head()
GS=GS["2004-02-20":]

BJK=BJK["2004-02-20":]
plt.plot(GS["Close"])

plt.plot(FB["Close"])
plt.plot(BJK["Close"])
plt.plot(BJK["Volume"])

FB['Daily Return']=FB['Close'].pct_change()



FB['Daily Return'].plot(figsize=(15,4),legend=True,linestyle='--',marker='o')

plt.ioff()



GS['Daily Return']=GS['Close'].pct_change()

GS['Daily Return'].plot(figsize=(15,4),legend=True,linestyle='--',marker='o')

plt.ioff()
FB['Daily Return'].hist(bins=50)

plt.ioff()
df=FB.index.copy()

df=pd.DataFrame(df)

df.index=df.DateTime

df['FB']=FB['Close']

df['GS']=GS['Close']

df['BJK']=BJK['Close']
df=df.drop("DateTime",axis=1)
sport_rets=df.pct_change()

sport_rets=pd.DataFrame(sport_rets)

sport_rets.shape
from scipy import stats

sns.jointplot('GS','FB',sport_rets,kind='scatter',color='seagreen').annotate(stats.pearsonr)

plt.ioff()
from scipy import stats

sns.jointplot('FB','BJK',sport_rets,kind='scatter',color='seagreen').annotate(stats.pearsonr)

plt.ioff()
from scipy import stats

sns.jointplot('GS','BJK',sport_rets,kind='scatter',color='seagreen').annotate(stats.pearsonr)

plt.ioff()
sns.heatmap(sport_rets.corr(),annot=True,cmap='summer',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)

plt.ioff()
rets=sport_rets.dropna()
area=np.pi*5

plt.scatter(rets.mean(),rets.std(),s=area)

plt.xlabel('Expected Return')

plt.ylabel('Risk')



for label, x,y in zip(rets.columns,rets.mean(),rets.std()):

    plt.annotate(

        label,

        xy=(x,y),xytext=(22,-3),

        textcoords='offset points',ha='right',va='bottom',

        arrowprops=dict(arrowstyle='-',connectionstyle='arc,rad=-0.3'))



for label,x,y in zip(rets.columns,rets.mean(),rets.std()):

    print(label + " Expected Return: " + str(round(x,5)) + " - Standard Deviation: " + str(round(y,3)))
print("If you invest in BJK, 95% of the times the worst daily loss will not exceed " + str(np.round(rets['BJK'].quantile(0.05),3)) + "%")

print("If you invest in GS, 95% of the times the worst daily loss will not exceed " + str(np.round(rets['GS'].quantile(0.05),3))+ "%")

print("If you invest in FB, 95% of the times the worst daily loss will not exceed " + str(np.round(rets['FB'].quantile(0.05),3))+ "%")