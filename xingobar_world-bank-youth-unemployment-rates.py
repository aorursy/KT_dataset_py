# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot

from plotly.graph_objs import Scatter, Figure, Layout

init_notebook_mode()

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/API_ILO_country_YU.csv')
df.head()
country = ['Canada','Australia','China','France','United Kingdom','Greece']
years = ['2010','2011','2012','2013','2014']

columns = ['Country','Year','Unemployment']

country_unemployment = pd.DataFrame(columns=columns)

for year in years:

    df[year] = df[year].map(lambda x:round(x,2))

#for country_name in df['Country Name'].unique():

for country_name in country:

    curr_country = df[df['Country Name'] == country_name]

    for year in years:

        values = curr_country[year].astype(np.float32).values[0]

        entry = pd.DataFrame([[country_name,year,values]],columns=columns)

        country_unemployment = country_unemployment.append(entry)

country_unemployment.head()
table_count = pd.pivot_table(data= country_unemployment,

                            index = ['Year'],

                            columns = ['Country'],

                            values=['Unemployment'],

                            aggfunc='mean')

ax = sns.heatmap(data = table_count['Unemployment'],vmin=0,annot=True,fmt='2.2f')

plt.title('Year vs Country')

ticks = plt.setp(ax.get_xticklabels(),rotation=45)
sns.set_style("whitegrid")

sns.set_color_codes('pastel')

ax = plt.subplot(111)

for country_name in country:

    curr = country_unemployment[country_unemployment['Country'] == country_name]

    #ax =sns.pointplot(data=curr,x='Year',y='Unemployment',ax=ax,label=country_name)

    curr.plot(x='Year',y='Unemployment',ax=ax,figsize=(8,6),label=country_name,marker='o')

    ax.set_ylabel('Unemployment rate')

plt.legend(loc='upper left',frameon=True,bbox_to_anchor=(1.05,1))

plt.title('Country Unemployment')
Europe = ['Germany','Denmark','France','Greece','United Kingdom','Finland',

          'Hungary','Albania','Austria','Belgium','Bulgaria',

         'Bosnia and Herzegovina','Belarus','Switzerland','Cyprus',

         'Czech Republic','Spain','Estonia','Iceland',

         'Latvia','Liechtenstein','Lithuania','Luxembourg']



columns = ['Country','Year','Unemployment']

all_country = df[df['Country Name'].isin(Europe)].reset_index(drop=True)

europe_df = pd.DataFrame(columns=columns)

for idx in range(len(all_country)):

    curr = all_country[['Country Name','2010','2011','2012','2013','2014']].iloc[idx]

    

    for year in years:

        #curr = curr[idx]

        values = curr[year]

        country = curr['Country Name']

        entry = pd.DataFrame([[country,year,values]],columns=columns)

        europe_df = europe_df.append(entry)

        

europe_df.head()
table_count = pd.pivot_table(data=europe_df,

                             index=['Year'],

                             columns=['Country'],

                             values=['Unemployment'],

                             aggfunc='mean')

fig,ax=plt.subplots(figsize=(8,6))

sns.heatmap(data=table_count['Unemployment'],

            vmin=0,annot=False,linewidth=.5,ax=ax)

plt.title('Europe vs Unemployment')