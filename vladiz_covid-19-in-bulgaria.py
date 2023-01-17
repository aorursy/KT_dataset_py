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
import geopandas as gpd

%matplotlib inline
events_df = pd.read_csv('/kaggle/input/covid19-measures-in-bulgaria/covid-19 Bulgaria measures.csv')
events_df
df = pd.read_csv('../input/covid19-ourworldindata/full_data.csv')
#leave only non zeros total cases
df = df[df['total_cases']!=0]
appear_date_df = df.groupby('location').agg({'date':['min','max']}).sort_values(by=('date','min'), ascending=True, na_position='last')
appear_date_df.columns = [' '.join(col).strip() for col in appear_date_df.columns.values]
df1_c = pd.merge(df, appear_date_df, on="location")
df1_c['days since first case'] = pd.to_datetime(df1_c['date']) - pd.to_datetime(df1_c['date min'])
df1_c['days since first case']=df1_c['days since first case']/np.timedelta64(1,'D')
#df1_c=df1_c.reset_index()
df1_c
kwargs = {"fontsize": 16,
          "fontstyle": "normal"
         }
#plot several countries

countries = ['Bulgaria',
             'Greece', 
             'Romania', 
             'Serbia', 
             'Macedonia', 
             'Turkey', 
             'Albania', 
             'Croatia',
             'Montenegro', 
             'Bosnia and Herzegovina', 
             'Slovenia',
             'Slovakia',
             'Moldova',
             'Hungary',
             'Kosovo'
            ]
df_plot = df1_c[ df1_c['location'].isin(countries)] 

df_balkans = df_plot.copy()

plt.figure(figsize=(20,12))
sns.lineplot(data=df_plot, x='days since first case',y='total_cases', hue='location').set(title = 'After the first case in each country on the Balkans')


df_annotate = df_plot.groupby('location').agg({'days since first case':'max', 'total_cases':  'max'})

for index, row in df_annotate.iterrows():
    plt.annotate(#'{}\n(day: {:.0f}, cases: {:.0f})'.format(row.name,row['days since first case'],row['total_cases']),
                row.name,
                 xy=(row['days since first case']+0.1,row['total_cases']), 
                 xytext=(row['days since first case']+0.5, row['total_cases']+3),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=0.5),
                **kwargs
                )
plt.show()


#plot Byulgaria and border countries

countries = ['Bulgaria',
             'Greece', 
             'Romania', 
             'Serbia', 
             'Macedonia', 
             #'Turkey'
            ]
df_plot = df1_c[ df1_c['location'].isin(countries)] 

plt.figure(figsize=(20,15))
sns.lineplot(data=df_plot, x='days since first case',y='total_cases', hue='location').set(title = 'Bulgaria and neighbour countries after the first case')


df_annotate = df_plot.groupby('location').agg({'days since first case':'max', 'total_cases':  'max'})

for index, row in df_annotate.iterrows():
    plt.annotate('{} (day: {:.0f}, cases: {:.0f})'.format(row.name,row['days since first case'],row['total_cases']),
                #row.name,
                 xy=(row['days since first case']+0.1,row['total_cases']), 
                 xytext=(row['days since first case']+0.5, row['total_cases']+50),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=0.5),
                **kwargs
                )
plt.show()
#plot Byulgaria and border countries

countries = ['Bulgaria',
             'Greece', 
             'Romania', 
             'Serbia', 
             'Macedonia', 
             #'Turkey'
            ]
df_plot = df1_c[ df1_c['location'].isin(countries)] 

plt.figure(figsize=(20,15))
sns.lineplot(data=df_plot, x='days since first case',y='new_cases', hue='location').set(title = 'New cases daily - Bulgaria and neighbour countries after the first case')


df_annotate = df_plot.groupby('location').agg({'days since first case':'max'})
df_annotate = pd.merge(df_annotate, df_plot, on=["location",'days since first case'], how = 'left')
print(df_annotate)

for index, row in df_annotate.iterrows():
    plt.annotate('{} (day: {:.0f}, cases: {:.0f})'.format(row.location,row['days since first case'],row['new_cases']),
                #row.name,
                 xy=(row['days since first case']+0.1,row['new_cases']), 
                 xytext=(row['days since first case']+0.5, row['new_cases']+10),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=0.5),
                **kwargs
                )
plt.show()
import datetime
datetime.datetime.strptime('2020-03-08', "%Y-%m-%d").date()

events_df = events_df[['date','restriction']]
events_df['first date'] = datetime.datetime.strptime('2020-03-08', "%Y-%m-%d").date()
events_df['days since first case'] = pd.to_datetime(events_df['date']) - pd.to_datetime(events_df['first date'])
events_df['days since first case']=events_df['days since first case']/np.timedelta64(1,'D')
events_df=events_df.reset_index()
#plot Byulgaria and events

countries = 'Bulgaria'
df_plot = df1_c[ df1_c['location'].eq(countries)] 

df_plot = pd.merge(df_plot, events_df, how='left', on=["days since first case",'date'])
#print(events_df)
plt.figure(figsize=(15,20))
sns.lineplot(data=df_plot, x='days since first case',y='total_cases', hue='location').set(title = 'Bulgaria after the first case')

# annotate with events
#print(df_plot.columns)

df_annotate = (df_plot.assign(rn=df_plot.groupby(['days since first case'])['total_cases']
                                          .rank(method='first', ascending=False))
                                          #.query('rn <= 2')
                                          .sort_values(['days since first case', 'rn']))
#print(df_annotate)
for index, row in df_annotate.iterrows():
    if str(row['restriction']) != str(np.nan):
        print(row['restriction'], np.nan)
        plt.annotate('day {:.0f}: {} - {}'.format(row['days since first case'],row['date'], row['restriction']),
                    #row.name,
                     xy=(row['days since first case']+0.1,row['total_cases']), 
                     xytext=(row['days since first case']+0.5, row['total_cases']-8*(row['rn'] - 1)),
                     arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=0.5),
                    **kwargs
                    )
plt.show()
countries = ['Bulgaria',
             'Greece', 
             'Romania', 
             'Serbia', 
             'Macedonia', 
             'Turkey', 
             'Albania', 
             'Croatia',
             'Montenegro', 
             'Bosnia and Herz.', 
             'Slovenia',
             'Slovakia',
             'Moldova',
             'Hungary',
             'Kosovo'
            ]

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
balkans = world[world.name.isin(countries)].sort_values('name')

df_balkans = df_balkans[df_balkans.date == df_balkans['date max']]

# manually change Bosnia and Herz. to Bosnia and Herzegovina
balkans.at[170,'name']= 'Bosnia and Herzegovina'

df_balkans = pd.merge(balkans, df_balkans, how='left', left_on='name', right_on='location')
df_balkans['coords'] = df_balkans['geometry'].apply(lambda x: x.representative_point().coords[:])
df_balkans['coords'] = [coords[0] for coords in df_balkans['coords']]
fig, ax = plt.subplots(1, figsize=(20, 15))
ax.axis('off')
ax.set_title('COVID-19 Total Cases by country in the Balkans', fontdict={'fontsize': '20', 'fontweight' : '3'})

#bbox_props = dict(boxstyle="round", fc="cyan", ec="b", lw=2)
bbox_props = dict(boxstyle="round",fc="white", ec="b" )

#df_balkans.plot(column='total_cases', cmap='Blues', figsize=(15,10), scheme='equal_interval', k=9, legend=True, alpha=0.5, edgecolor='k')
ax = df_balkans.plot(column='total_cases', cmap='Blues', edgecolor='k', ax=ax)
df_balkans.apply(lambda x: ax.annotate(s='{}\n{:.0f}'.format(x.location, x.total_cases), xy=x.geometry.centroid.coords[0], ha='center', bbox=bbox_props),axis=1)
#df_balkans
