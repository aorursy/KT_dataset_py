import pandas as pd
import numpy as np
import seaborn as sns
flow_age_201902 = pd.read_csv('/kaggle/input/flow-age/4_FLOW_AGE_201902.CSV',sep='|')
flow_age_201903 = pd.read_csv('/kaggle/input/flow-age/4_FLOW_AGE_201903.CSV',sep='|')
flow_age_201904 = pd.read_csv('/kaggle/input/flow-age/4_FLOW_AGE_201904.CSV',sep='|')
flow_age_201905 = pd.read_csv('/kaggle/input/flow-age/4_FLOW_AGE_201905.CSV',sep='|')
flow_age_202002 = pd.read_csv('/kaggle/input/flow-age/4_FLOW_AGE_202002.CSV',sep='|')
flow_age_202003 = pd.read_csv('/kaggle/input/flow-age/4_FLOW_AGE_202003.CSV',sep='|')
flow_age_202004 = pd.read_csv('/kaggle/input/flow-age/4_FLOW_AGE_202004.CSV',sep='|')
flow_age_202005 = pd.read_csv('/kaggle/input/flow-age/4_FLOW_AGE_202005.CSV',sep='|')

flow_age = flow_age_201902.append(flow_age_201903)
flow_age = flow_age.append(flow_age_201904)
flow_age = flow_age.append(flow_age_201905)
flow_age = flow_age.append(flow_age_202002)
flow_age = flow_age.append(flow_age_202003)
flow_age = flow_age.append(flow_age_202004)
flow_age = flow_age.append(flow_age_202005)
flow_age=flow_age.groupby('STD_YMD').sum().loc[:,'MAN_FLOW_POP_CNT_0004':].T
flow_age.index.name = 'age_category'
flow_age.columns.name = None
flow_age.head()
flow_age.index
flow_age_man = flow_age.loc[:'MAN_FLOW_POP_CNT_70U']
flow_age_woman = flow_age.loc['WMAN_FLOW_POP_CNT_0509':]
flow_age_man=flow_age_man.T
flow_age_man.columns.name=None
flow_age_man.index.name='YM'
flow_age_woman = flow_age_woman.T
flow_age_woman.columns.name=None
flow_age_woman.index.name='YM'
flow_age_man
flow_age_man.iloc[142]
flow_age_woman=flow_age_woman.reset_index()
flow_age_man=flow_age_man.reset_index()
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(32,16))
sns.lineplot(flow_age_man.index[:120],flow_age_man.iloc[:120,1:].sum(axis=1),color='blue',label='2019',linewidth=4)
sns.lineplot(flow_age_man.index[120:]-120,flow_age_man.iloc[120:,1:].sum(axis=1),color='orange',label='2020',linewidth=4)
plt.axvline(x=22,color='red') # 신천지 코로나 확산 (확진자 세자리)
plt.title('Male 2019 02 ~ 05, 2020 02 ~ 05 flow graph',fontsize=40)
plt.legend(fontsize=30)
plt.xticks(flow_age_man.index[:120],flow_age_man.YM)
plt.xticks(rotation=70,fontsize=11)
plt.text(24,2000000,'02.24 Shincheonji',fontsize=30,color='red')
plt.show() 

plt.figure(figsize=(32,16))
sns.lineplot(flow_age_woman.index[:120],flow_age_woman.iloc[:120,1:].sum(axis=1),color='blue',linewidth=4)
sns.lineplot(flow_age_woman.index[120:]-120,flow_age_woman.iloc[120:,1:].sum(axis=1),color='orange',linewidth=4)
plt.axvline(x=22,color='red') # 신천지 코로나 확산 (확진자 세자리)
plt.title('Female 2019 02 ~ 05, 2020 02 ~ 05 flow graph',fontsize=40)
plt.legend(fontsize=30)
plt.xticks(flow_age_woman.index[:120],flow_age_woman.YM)
plt.xticks(rotation=70,fontsize=11)
plt.text(24,2000000,'02.24 Shincheonji',fontsize=30,color='red')
plt.show() 

plt.figure(figsize=(32,16))
sns.lineplot(flow_age_man.index[:120],flow_age_man.iloc[:120,1:4].mean(axis=1),color='blue',label='2019 age 00~14',linewidth=3)
sns.lineplot(flow_age_man.index[120:]-120,flow_age_man.iloc[120:,1:4].mean(axis=1),color='orange',label='2020 age 00~14',linewidth=3)
sns.lineplot(flow_age_man.index[:120],flow_age_man.iloc[:120,4:13].mean(axis=1),color='yellow',label='2019 age 15~59',linewidth=3)
sns.lineplot(flow_age_man.index[120:]-120,flow_age_man.iloc[120:,4:13].mean(axis=1),color='red',label='2020 age 15~59',linewidth=3)
sns.lineplot(flow_age_man.index[:120],flow_age_man.iloc[:120,13:].mean(axis=1),color='black',label='2019 age 60~',linewidth=3)
sns.lineplot(flow_age_man.index[120:]-120,flow_age_man.iloc[120:,13:].mean(axis=1),color='green',label='2020 age 60~',linewidth=3)
plt.axvline(x=22,color='red') # 신천지 코로나 확산 (확진자 세자리)
plt.title('Male 2019 02 ~ 05, 2020 02 ~ 05 flow graph age category',fontsize=40)
plt.legend(fontsize=15)
plt.text(24,50000,'02.24 Shincheonji',fontsize=30,color='red')
plt.xticks(flow_age_man.index[:120],flow_age_man.YM)
plt.xticks(rotation=70,fontsize=11)
plt.show() 
plt.figure(figsize=(32,16))
sns.lineplot(flow_age_woman.index[:120],flow_age_woman.iloc[:120,1:3].mean(axis=1),color='blue',label='2019 age 5~14',linewidth=3)
sns.lineplot(flow_age_woman.index[120:]-120,flow_age_woman.iloc[120:,1:3].mean(axis=1),color='orange',label='2020 age 5~14',linewidth=3)
sns.lineplot(flow_age_woman.index[:120],flow_age_woman.iloc[:120,3:12].mean(axis=1),color='yellow',label='2019 age 15~59',linewidth=3)
sns.lineplot(flow_age_woman.index[120:]-120,flow_age_woman.iloc[120:,3:12].mean(axis=1),color='purple',label='2020 age 15~59',linewidth=3)
sns.lineplot(flow_age_woman.index[:120],flow_age_woman.iloc[:120,12:].mean(axis=1),color='black',label='2019 age 60~',linewidth=3)
sns.lineplot(flow_age_woman.index[120:]-120,flow_age_woman.iloc[120:,12:].mean(axis=1),color='brown',label='2020 age 60~',linewidth=3)
plt.axvline(x=22,color='red') # 신천지 코로나 확산 (확진자 세자리)
plt.title('Female 2019 02 ~ 05, 2020 02 ~ 05 flow graph age category',fontsize=40)
plt.legend(fontsize=15)
plt.xticks(flow_age_woman.index[:120],flow_age_woman.YM)
plt.xticks(rotation=70,fontsize=11)
plt.text(24,50000,'02.24 Shincheonji',fontsize=30,color='red')
plt.show() 
flow_age_man.drop(flow_age_man[flow_age_man['YM']==20200229].index,inplace=True)
flow_age_woman.drop(flow_age_woman[flow_age_woman['YM']==20200229].index,inplace=True)
#Drop 20200229 row data to compare 2019 2020
flow_age_man_2019=flow_age_man.iloc[:120]
flow_age_man_2020=flow_age_man.iloc[120:]
flow_age_woman_2019=flow_age_woman.iloc[:120]
flow_age_woman_2020=flow_age_woman.iloc[120:]