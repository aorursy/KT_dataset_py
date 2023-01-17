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
%matplotlib inline
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.head()
df.tail()
df.info()
df.describe()
min(df['ObservationDate']) , max(df['ObservationDate'])
df[df['ObservationDate'] == max(df['ObservationDate'])]
my_df = df[df['ObservationDate'] == max(df['ObservationDate'])][['ObservationDate','Country/Region','Last Update','Confirmed','Deaths','Recovered']]
my_df
#my_df.groupby(['ObservationDate','Country/Region','Last Update']).agg({'Confirmed': ['mean', 'sum'], 'Deaths': ['mean', 'sum'], 'Recovered': ['mean', 'sum']})

df_agg = my_df.groupby(['ObservationDate','Country/Region']).agg({'Last Update':'max', 'Confirmed':  'sum', 'Deaths': 'sum', 'Recovered':  'sum'}).sort_values(by='Confirmed', ascending=False, na_position='first')

#df_agg = my_df.groupby(['ObservationDate','Country/Region']).agg({'Last Update':'max', 'Confirmed':  'sum', 'Deaths': 'sum', 'Recovered':  'sum'}).sort_values(by='Confirmed', ascending=False, na_position='first')
#my_df = df[df['ObservationDate'] == max(df['ObservationDate'])][['ObservationDate','Country/Region','Last Update','Confirmed','Deaths','Recovered']]

appearance_df = df.groupby('Country/Region').agg({'ObservationDate':['min','max']}).sort_values(by=('ObservationDate','min'), ascending=True, na_position='last')
appearance_df.columns = [' '.join(col).strip() for col in appearance_df.columns.values]
appearance_df
# manually add the first case date
appearance_df.at['Mainland China','ObservationDate min']= '11/17/2019'

appearance_df
df_agg.head(10)
df_agg['Active'] = df_agg['Confirmed'] - df_agg['Deaths'] - df_agg['Recovered']
df_agg['Mortality Rate'] = df_agg['Deaths']/df_agg['Confirmed']
df_agg['Recovery Rate'] = df_agg['Recovered']/df_agg['Confirmed']
df_agg['Active Rate'] = df_agg['Active']/df_agg['Confirmed']
df_agg.head(10)
df_c = pd.merge(df_agg, appearance_df, how='left', on="Country/Region")
df_c
df_c['days_in_country'] = pd.to_datetime(df_c['ObservationDate max']) - pd.to_datetime(df_c['ObservationDate min'])
df_c['days_in_country']=df_c['days_in_country']/np.timedelta64(1,'D')
df_c['AVG cases per day'] = df_c['Confirmed']/df_c['days_in_country']
df_c.head(10)
df_1 = df[['ObservationDate','Country/Region','Last Update','Confirmed','Deaths','Recovered']]
df1_agg = df_1.groupby(['Country/Region','ObservationDate']).agg({'Last Update':'max', 'Confirmed':  'sum', 'Deaths': 'sum', 'Recovered':  'sum'}).sort_values(by=['Country/Region','ObservationDate'], ascending=True, na_position='first')
#df1_agg
df1_agg['Active'] = df1_agg['Confirmed'] - df1_agg['Deaths'] - df1_agg['Recovered']
df1_agg['Mortality Rate'] = df1_agg['Deaths']/df1_agg['Confirmed']
df1_agg['Recovery Rate'] = df1_agg['Recovered']/df1_agg['Confirmed']
df1_agg['Active Rate'] = df1_agg['Active']/df1_agg['Confirmed']
df1_agg.reset_index(level='ObservationDate', inplace=True)
#df1_agg

df1_c = pd.merge(df1_agg, appearance_df, on="Country/Region")
# here is the change 'ObservationDate' - 'ObservationDate min' instaed of 'ObservationDate min' - 'ObservationDate min' because it is different for each day
df1_c['days_in_country'] = pd.to_datetime(df1_c['ObservationDate']) - pd.to_datetime(df1_c['ObservationDate min'])
df1_c['days_in_country']=df1_c['days_in_country']/np.timedelta64(1,'D')
df1_c['AVG cases per day'] = df1_c['Confirmed']/df1_c['days_in_country']
df1_c
df1_c=df1_c.reset_index()
#plot several countries

#countries = ['Bulgaria','Italy', 'Spain', 'Germany', 'Switzerland', 'France', 'Iran']
countries = ['Bulgaria',
             'Greece', 
             'Romania', 
             'Serbia', 
             'North Macedonia', 
             #'Turkey', 
             'Albania', 
             'Croatia',
             'Montenegro', 
             'Bosnia and Herzegovina', 
             'Slovenia',
             'Slovakia',
             'Moldova',
             'Hungary'
            ]
df_plot = df1_c[ df1_c['Country/Region'].isin(countries)] 
#['Country/Region','days_in_country','Confirmed'] 

plt.figure(figsize=(20,9))
sns.lineplot(data=df_plot, x='days_in_country',y='Confirmed', hue='Country/Region').set(title = 'COVID-19 around the Balkans, Confirmed Cases')


df_annotate = df_plot.groupby('Country/Region').agg({'days_in_country':'max', 'Confirmed':  'max'})

for index, row in df_annotate.iterrows():
    #print(row.name)
    plt.annotate(#'{}\n(day: {:.0f}, cases: {:.0f})'.format(row.name,row['days_in_country'],row['Confirmed']),
                row.name,
                 xy=(row['days_in_country']+0.1,row['Confirmed']), 
                 xytext=(row['days_in_country']+0.5, row['Confirmed']+3),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=0.5))
plt.show()

#plot Active cases ####################################################################################################
plt.figure(figsize=(20,9))
sns.lineplot(data=df_plot, x='days_in_country',y='Active', hue='Country/Region').set(title = 'COVID-19 around the Balkans, Active Cases')


df_annotate = df_plot.groupby('Country/Region').agg({'days_in_country':'max'})
print(df_annotate)
df_annotate = pd.merge(df_annotate, df_plot,on=['Country/Region','days_in_country'], how='left' )

for index, row in df_annotate.iterrows():
    #print(row.name)
    plt.annotate(#'{}\n(day: {:.0f}, cases: {:.0f})'.format(row.name,row['days_in_country'],row['Confirmed']),
                row['Country/Region'],
                 xy=(row['days_in_country']+0.1,row['Active']), 
                 xytext=(row['days_in_country']+0.5, row['Active']+3),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=0.5))
plt.show()
#plot only for Bulgaria

#countries = ['Bulgaria','Italy', 'Spain', 'Germany', 'Switzerland', 'France', 'Iran']
countries = ['Bulgaria']

df_plot = df1_c[ df1_c['Country/Region'].isin(countries)] 
#['Country/Region','days_in_country','Confirmed'] 

plt.figure(figsize=(20,9))
sns.lineplot(data=df_plot, x='ObservationDate',y='Confirmed').set(title = 'COVID-19 in Bulgaria')
plt.xticks(rotation=45)

df_annotate = df_plot.groupby('Country/Region').agg({'days_in_country':'max', 'Confirmed':  'max'})

for index, row in df_annotate.iterrows():
    #print(row.name)
    plt.annotate(#'{}\n(day: {:.0f}, cases: {:.0f})'.format(row.name,row['days_in_country'],row['Confirmed']),
                'Total confirmed',
                 xy=(row['days_in_country']+0.1,row['Confirmed']), 
                 xytext=(row['days_in_country']+0.5, row['Confirmed']+3),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=0.5))
#plt.show()

#plot Active cases ####################################################################################################
#plt.figure(figsize=(20,9))
#Plotting on the same chart

sns.lineplot(data=df_plot, x='ObservationDate',y='Active')


df_annotate = df_plot.groupby('Country/Region').agg({'days_in_country':'max'})
print(df_annotate)
df_annotate = pd.merge(df_annotate, df_plot,on=['Country/Region','days_in_country'], how='left' )

for index, row in df_annotate.iterrows():
    #print(row.name)
    plt.annotate(#'{}\n(day: {:.0f}, cases: {:.0f})'.format(row.name,row['days_in_country'],row['Confirmed']),
                'Active',
                 xy=(row['days_in_country']+0.1,row['Active']), 
                 xytext=(row['days_in_country']+0.5, row['Active']+3),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=0.5))
plt.show()
