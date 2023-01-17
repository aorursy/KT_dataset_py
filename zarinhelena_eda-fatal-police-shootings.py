import os 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from datetime import date
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/police-deadly-force-usage-us/fatal-police-shootings-data.csv', delimiter=',')
df.head()
df.info()
df.nunique()
df['date'] = pd.to_datetime(df['date'])

df['year'] = df.date.dt.year

df.head()
df.gender.value_counts()
df[df.gender.isna()]
df.gender.fillna(value='Unknown',inplace=True)

df.gender.value_counts()
plt.figure(figsize=(8,5))

ax = sns.countplot(x='gender', data=df);

total = df.gender.value_counts().sum()

for p in ax.patches:

    perc = '{:.2f}%'.format(100 * p.get_height()/total)

    ax.annotate(perc, (p.get_x()+0.33, p.get_height()+50))

plt.title('Police Shooting Deaths in US (2015-2020)', fontsize=16)

plt.show()
df_M=df[df['gender']=='M'].groupby(['year'])['id'].count().reset_index()

df_F=df[df['gender']=='F'].groupby(['year'])['id'].count().reset_index()



fig = plt.figure(figsize=(12,5))

g = GridSpec(1,2)

ax1 = fig.add_subplot(g[0,0])

ax2 = fig.add_subplot(g[0,1])



sns.barplot(x='year', y='id', ax=ax1, data=df_M)

sns.barplot(x='year', y='id', ax=ax2, data=df_F)

ax1.set_title('Male population')

ax2.set_title('Female population')

ax1.set_ylabel('count')

ax2.set_ylabel('count')

plt.show()
df.race.notna().value_counts()
df.race.fillna(value='U',inplace=True)

df.race.value_counts()
race_grouped = df.groupby(['year','race'])['id'].count().unstack(level=-1)

race_grouped = race_grouped[['W', 'B', 'H', 'U', 'A', 'N', 'O']]

race_grouped = race_grouped.rename(columns={'W': 'White', 'B': 'Black', 'H': 'Hispanic', 

                    'U': 'Unknown', 'A': 'Asian', 'N': 'Native American', 'O': 'Other'})

race_grouped
plt.figure(figsize=(12,6))

race_grouped.plot.barh(stacked=True)

plt.legend(bbox_to_anchor=(1.1, 1.05))

plt.show()
state_pops = pd.read_csv('../input/us-state-populations-2018/State Populations.csv')

state_codes = {'California' : 'CA', 'Texas' : 'TX', 'Florida' : 'FL', 'New York' : 'NY', 'Pennsylvania' : 'PA',

       'Illinois' : 'IL', 'Ohio' : 'OH', 'Georgia' : 'GA', 'North Carolina' : 'NC', 'Michigan' : 'MI',

       'New Jersey' : 'NJ', 'Virginia' : 'VA', 'Washington' : 'WA', 'Arizona' : 'AZ', 'Massachusetts' : 'MA',

       'Tennessee' : 'TN', 'Indiana' : 'IN', 'Missouri' : 'MO', 'Maryland' : 'MD', 'Wisconsin' : 'WI',

       'Colorado' : 'CO', 'Minnesota' : 'MN', 'South Carolina' : 'SC', 'Alabama' : 'AL', 'Louisiana' : 'LA',

       'Kentucky' : 'KY', 'Oregon' : 'OR', 'Oklahoma' : 'OK', 'Connecticut' : 'CT', 'Iowa' : 'IA', 'Utah' : 'UT',

       'Nevada' : 'NV', 'Arkansas' : 'AR', 'Mississippi' : 'MS', 'Kansas' : 'KS', 'New Mexico' : 'NM',

       'Nebraska' : 'NE', 'West Virginia' : 'WV', 'Idaho' : 'ID', 'Hawaii' : 'HI', 'New Hampshire' : 'NH',

       'Maine' : 'ME', 'Montana' : 'MT', 'Rhode Island' : 'RI', 'Delaware' : 'DE', 'South Dakota' : 'SD',

       'North Dakota' : 'ND', 'Alaska' : 'AK', 'District of Columbia' : 'DC', 'Vermont' : 'VT',

       'Wyoming' : 'WY'}

state_pops['State Code'] = state_pops['State'].apply(lambda x: state_codes[x])

state_pops.head()
state_grouped = df.groupby(['state'])['id'].count().reset_index()

state_grouped.head()
state_grouped['population'] = state_grouped['state'].apply(lambda x: state_pops[state_pops['State Code'] == x].values[0,1])

state_grouped['per100000'] = state_grouped['id']/state_grouped['population']*100000

state_grouped.sort_values('per100000', ascending=False, inplace=True)

state_grouped.head()
plt.figure(figsize=(8,15))

sns.barplot(x='per100000', y='state', data=state_grouped)

plt.title('Police Shootings in US (2015-2020)', fontsize=16)

plt.xlabel('Deaths per 100,000')

plt.show()