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
%matplotlib inline
df = pd.read_csv('../input/covid19-ourworldindata/full_data.csv')
df_copy = df.copy()
df.head()
#filter data only with zeros
df = df[df['total_cases']!=0]
#df.head()
appear_date_df = df.groupby('location').agg({'date':['min','max']}).sort_values(by=('date','min'), ascending=True, na_position='last')
appear_date_df.columns = [' '.join(col).strip() for col in appear_date_df.columns.values]
appear_date_df.head()
df1_c = pd.merge(df, appear_date_df, on="location")
#df1_c
df1_c['days since first case'] = pd.to_datetime(df1_c['date']) - pd.to_datetime(df1_c['date min'])
df1_c['days since first case']=df1_c['days since first case']/np.timedelta64(1,'D')
#df1_c
df1_c=df1_c.reset_index()

kwargs = {"fontsize": 12,
          "fontstyle": "normal"
         }
#plot several countries

countries = ['Bulgaria',
             'Greece', 
             'Romania', 
             'Serbia', 
             'North Macedonia', 
             'Macedonia',
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
df_plot = df1_c[ df1_c['location'].isin(countries)] 

plt.figure(figsize=(20,12))
sns.lineplot(data=df_plot, x='days since first case',y='total_cases', hue='location').set(title = 'COVID-19 around the Balkans (after first case)')


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
#filter data above 100 total_cases
df=df_copy.copy()
df = df[df['total_cases']>=100]
df.head()

appear_date_df = df.groupby('location').agg({'date':['min','max']}).sort_values(by=('date','min'), ascending=True, na_position='last')
appear_date_df.columns = [' '.join(col).strip() for col in appear_date_df.columns.values]
df1_c = pd.merge(df, appear_date_df, on="location")
df1_c['days since 100th case'] = pd.to_datetime(df1_c['date']) - pd.to_datetime(df1_c['date min'])
df1_c['days since 100th case']=df1_c['days since 100th case']/np.timedelta64(1,'D')
df1_c=df1_c.reset_index()

#plot several countries

countries = ['Bulgaria',
             'Greece', 
             'Romania', 
             'Serbia', 
             'North Macedonia', 
             'Macedonia',
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
df_plot = df1_c[ df1_c['location'].isin(countries)] 

plt.figure(figsize=(20,12))
sns.lineplot(data=df_plot, x='days since 100th case',y='total_cases', hue='location').set(title = 'COVID-19 around the Balkans (after 100th case)')

df_annotate = df_plot.groupby('location').agg({'days since 100th case':'max', 'total_cases':  'max'})

for index, row in df_annotate.iterrows():
    plt.annotate(row.name,
                 xy=(row['days since 100th case']+0.1,row['total_cases']), 
                 xytext=(row['days since 100th case']+0.5, row['total_cases']+3),
                 arrowprops=dict(facecolor='black', shrink=0.05, headwidth=4, width=0.5),
                **kwargs)
plt.show()
