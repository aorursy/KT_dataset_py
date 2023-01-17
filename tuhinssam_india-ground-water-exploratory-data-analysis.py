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

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/districtwise-ground-water-resources-by-july-2017/Dynamic_2017_2_0.csv')
df.shape
df.sample(10)
df.drop('S.no.', axis =1, inplace = True)

f, ax = plt.subplots(figsize=(10,7))

sns.heatmap(df.corr(), annot = True)

plt.show()
#Get information on the columns, null values in the data

df.info()
df.isna().sum()
#statewise groundwater reserve

state_list = []

total_frnd_water_recharge = []

curr_gw_extr_list = []

future_available_GW_list = []

#Net Ground Water Availability for future use



for state, subset in df.groupby('Name of State'):

    #print(state, sum(subset['Net Ground Water Availability for future use']))

    state_list.append(state)

    total_frnd_water_recharge.append(sum(subset['Total Annual Ground Water Recharge']))

    curr_gw_extr_list.append(sum(subset['Total Current Annual Ground Water Extraction']))

    future_available_GW_list.append(sum(subset['Net Ground Water Availability for future use'])) 

dfnew = pd.DataFrame({"State":state_list, "GW_Recharge":total_frnd_water_recharge, "GW_Extraction": curr_gw_extr_list, "Future_GW_Available": future_available_GW_list})
dfnew.sort_values(['GW_Recharge','GW_Extraction'], inplace= True)

dfnew['annual_reserve'] = dfnew['GW_Recharge']-dfnew['GW_Extraction']

dfnew
f, ax = plt.subplots(figsize=(10,8))

plt.barh(dfnew['State'],dfnew['annual_reserve'], color=(dfnew['annual_reserve']>0).map({True: 'g',False: 'r'}))

plt.show()
f, ax = plt.subplots(figsize=(10, 9))



sns.set_color_codes("muted")

sns.barplot(x='GW_Recharge', y= 'State', data = dfnew, label = 'Available Ground Water', color='b')

sns.barplot(x='GW_Extraction', y= 'State', data = dfnew, label = 'GroundWater Extraction', color='r')

ax.legend(ncol=2, loc="lower right", frameon=True)

plt.show()
dfnew.sort_values('Future_GW_Available', inplace = True)

f, ax = plt.subplots(figsize=(10, 9))

sns.barplot(x='Future_GW_Available', y = 'State',data = dfnew)

plt.show()
dfnew.count()
df_westbengal = df[df['Name of State']=='WEST BENGAL']

df_westbengal.head()
for state in list(df['Name of State'].unique()):

    print(state)


sns.barplot(x = 'Annual GW Allocation for Domestic Use as on 2025', y = 'Name of District', data = df[df['Name of State']=='WEST BENGAL'])

plt.show()
df_westbengal.sort_values('Stage of Ground Water Extraction (%)',ascending=False, inplace = True)

df_westbengal[['Name of State','Name of District','Stage of Ground Water Extraction (%)']].head(3)
plt.figure(figsize=(10,12))

sns.barplot(x='Recharge from rainfall During Monsoon Season', y='Name of State', data = df)

plt.show()
fig,axs = plt.subplots(6,6,figsize=(12,8))



for idx,state in enumerate(df['Name of State'].value_counts().sort_values(ascending=False)[0:36].index):

    print(idx,state)

    axs[idx//6,idx%6].hist(x = df[df['Name of State']==state]['Stage of Ground Water Extraction (%)'], color='b')

    axs[idx//6,idx%6].set_title(state)

plt.suptitle("State wise GW Extraction distribution")

plt.tight_layout()

fig.subplots_adjust(top=0.88)

plt.show()