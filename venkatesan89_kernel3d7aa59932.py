# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('/kaggle/input/uncover/UNCOVER/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv')
df.head()
df.shape
df[df['countryterritorycode']=='IND'].head()
df['countriesandterritories'].value_counts()
df['year'].value_counts()
df[df['countryterritorycode']=='IND'].shape
df[df['countryterritorycode']=='IND'].sort_values(['year', 'month','day'], ascending=[0,0,0 ])
df[(df['countryterritorycode']=='IND') & (df['month']==1)].sort_values(['year', 'month','day'], ascending=[0,0,0 ])
df_india=df[df['countryterritorycode']=='IND'].copy()
df_india.head()
df_india.shape
df_india['day_str']=df_india['day'].apply(lambda x:str(x))
df_india['day_month']=df_india['month'].apply(lambda x:str(x))
df_india['day_year']=df_india['year'].apply(lambda x:str(x))
df_india["timestamp"] = (df_india['day_str']).str.cat(df_india['day_month'],sep=" ").str.cat(df_india['day_year'],sep=" ")
df_india.drop(columns=['day_str','day_month','day_year'],inplace=True)
import datetime
df_india["timestamp"] = df_india['timestamp'].apply(lambda x:datetime.datetime.strptime(x, '%d %m %Y'))
df_india.head()
df_india=df_india.sort_values(['timestamp'], ascending=True)
cum_case=0
cum_death=0
list_cum_case=[]
list_cum_death=[]
for i in range(5922,5803,-1):
    #print(i)
    #print(df_india.loc[i]['cases'])
    cum_case+=df_india.loc[i]['cases']
    list_cum_case.append(cum_case)
    cum_death+=df_india.loc[i]['deaths']
    list_cum_death.append(cum_death)
    
    
df_india['cum_case']=list_cum_case
df_india['cum_death']=list_cum_death
df_india
fig=plt.figure(figsize=(20,5))
plt.plot(df_india['timestamp'], df_india['cum_case'], 'bo--')
fig=plt.figure(figsize=(20,5))
plt.plot(df_india['timestamp'], df_india['cum_death'], 'ro--')
df_india['death_by_case_perc']=df_india['cum_death']*100/df_india['cum_case']
df_india['death_by_case_perc'].fillna(0,inplace=True)
fig=plt.figure(figsize=(20,5))
plt.plot(df_india['timestamp'], df_india['death_by_case_perc'], 'go--')
df_bycountries=df.groupby('countriesandterritories')[['cases','deaths']].sum().sort_values('cases',ascending=False).head(30)
df_bycountries1=df_bycountries.reset_index()
import seaborn as sns
fig=plt.figure(figsize=(40,18))
plt.subplot(2,1,1)
sns.barplot(x="countriesandterritories", y="cases", data=df_bycountries1)
plt.title('Number of cases in different countries')
plt.subplot(2,1,2)
sns.barplot(x="countriesandterritories", y="deaths", data=df_bycountries1.sort_values('deaths',ascending=False))
plt.title('Number of deaths in different countries')
df_bycountries1['perc_death']=df_bycountries1['deaths']*100/df_bycountries1['cases']
fig=plt.figure(figsize=(40,8))
sns.barplot(x="countriesandterritories", y="perc_death", data=df_bycountries1.sort_values('perc_death',ascending=False))
plt.title('Death percentage to cases in different countries')
df_bycountries=df.groupby('countriesandterritories')['popdata2018'].max().reset_index()
df_bycountries = pd.merge(df_bycountries1, df_bycountries, on='countriesandterritories')
df_bycountries['perc_cases_pop']=df['cases']*100/df['popdata2018']
df_bycountries['perc_deaths_pop']=df['deaths']*100/df['popdata2018']
df_bycountries.head(9)
fig=plt.figure(figsize=(40,16))
plt.subplot(2,1,1)
sns.barplot(x="countriesandterritories", y="perc_cases_pop", data=df_bycountries.sort_values('perc_cases_pop',ascending=False))
plt.title('Cases percentage to population in different countries')
plt.subplot(2,1,2)
sns.barplot(x="countriesandterritories", y="perc_deaths_pop", data=df_bycountries.sort_values('perc_deaths_pop',ascending=False))
plt.title('Death percentage to population in different countries')