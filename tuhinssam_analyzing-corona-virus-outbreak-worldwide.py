from IPython.display import YouTubeVideo

YouTubeVideo('mOV1aBVYKGA', width=800, height=300)
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
import matplotlib.pyplot as plt

import seaborn as sns
df_corona = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv",index_col='Last Update', parse_dates=['Last Update'])

df_corona.head(10)
df_corona = df_corona.head(497)
df_corona[df_corona['Province/State']=='Hubei']
df_corona.info()
df_corona.isnull().sum()
plt.figure(figsize = (16,8))

sns.set_style("darkgrid")

g = sns.lineplot(data =df_corona['Confirmed'], label = "Confirmed Cases")

sns.lineplot(data =df_corona['Deaths'], label = "Deaths")

sns.lineplot(data =df_corona['Recovered'], label = "Recovered")

plt.xticks(rotation=45)

plt.show()
df_cases_wo_china = df_corona[(df_corona['Country'] != "China") & (df_corona['Country'] != "Mainland China")]

country_conf_max = []

country_reco_max = []

country_deth_max = []

country_val = []

for country, subset in df_cases_wo_china.groupby('Country'):

    country_conf_max.append(max(subset['Confirmed']))

    country_reco_max.append(max(subset['Recovered'])) 

    country_deth_max.append(max(subset['Deaths']))

    country_val.append(country)

df_country_woc = pd.DataFrame({"Country": country_val, "Confirmed":country_conf_max, "Recovered": country_reco_max, "Death": country_deth_max})

df_woc = df_country_woc.sort_values('Confirmed', ascending = False)

df_woc_top10 = df_woc.head(10)
df_woc[df_woc['Confirmed']>0][['Country','Confirmed','Recovered', "Death"]]
#Top 10 Corona Virus affed Country Outside China

df_woc_top10
plt.figure(figsize=(10,6))

sns.barplot(x='Confirmed', y= 'Country', data = df_woc_top10, color='r', label='Confirmed Cases')

sns.barplot(x='Recovered', y= 'Country', data = df_woc_top10, color='g', label='Recovered')

plt.legend()

plt.show()
df_recovered_osc = df_woc_top10.sort_values('Recovered', ascending= False).head(5)

sns.set_style('whitegrid')

plt.figure(figsize=(6,5))

sns.barplot(x=df_recovered_osc['Country'], y=df_recovered_osc['Recovered'])

plt.xticks(rotation=60)

plt.show()
df_corona_china = df_corona[(df_corona['Country'] == "China") | (df_corona['Country'] == "Mainland China")]

df_corona_china
for Country, subset in df_corona_china.groupby('Country'):

    print(Country,max(subset['Confirmed']))
st_conf_max = []

st_reco_max = []

st_deth_max = []

state_val = []

for state, subset in df_corona_china.groupby('Province/State'):

    st_conf_max.append(max(subset['Confirmed']))

    st_reco_max.append(max(subset['Recovered'])) 

    st_deth_max.append(max(subset['Deaths']))

    state_val.append(state)

df_china_bystate = pd.DataFrame({"State": state_val, "Confirmed":st_conf_max, "Recovered": st_reco_max, "Death": st_deth_max})

df_china_bystate = df_china_bystate.sort_values('Confirmed', ascending = False)

df_china_bystate_top10 = df_china_bystate.head(10)
df_china_bystate_top10
plt.figure(figsize=(12,6))

sns.barplot(y='State', x= 'Confirmed', data = df_china_bystate_top10, color='r',label='China Confirmed')

sns.barplot(y='State', x= 'Recovered', data = df_china_bystate_top10, color='g', label='Recovered')

plt.legend()

plt.show()
df_corona.sort_index(inplace = True)

plt.figure(figsize = (16,6))

sns.set_style("whitegrid")

sns.lineplot(data =df_corona[df_corona['Province/State']=='Hubei']['Confirmed'], label = "Hubei Confirmed Cases")

sns.lineplot(data =df_corona[df_corona['Province/State']=='Zhejiang']['Confirmed'], label = "Zhejiang Confirmed Cases")

sns.lineplot(data =df_corona[df_corona['Province/State']=='Guangdong']['Confirmed'], label = "Guangdong Confirmed Cases")

plt.xticks(rotation=45)

plt.show()
df_corona.sort_index(inplace = True)

plt.figure(figsize = (16,8))

sns.set_style("whitegrid")

sns.lineplot(data =df_corona[df_corona['Country']=='Mainland China']['Confirmed'], label = "Mainland China Confirmed Cases")

sns.lineplot(data =df_corona[df_corona['Country']=='China']['Confirmed'], label = "China Confirmed Cases")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize = (16,6))

sns.set_style("whitegrid")

sns.lineplot(data =df_corona[(df_corona['Country']!='Mainland China') & (df_corona['Country']!='China')]['Confirmed'], label = "Outside China Confirmed Cases", color = 'r')

plt.xticks(rotation=45)

plt.show()