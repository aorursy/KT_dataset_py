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
# libraries



import pandas as pd

import numpy as np

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.style as style

style.use('fivethirtyeight')

import seaborn as sns

from pylab import rcParams

rcParams['figure.figsize'] = 14, 8.7 

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

df.head()
df.shape
# Number of each type of column



df.dtypes.value_counts()
# unique values of object types



df.select_dtypes('object').apply(lambda x: pd.Series.nunique(x))
# unique values of float types



df.select_dtypes('float64').apply(lambda x: pd.Series.nunique(x))
# unique values of int types



df.select_dtypes('int64').apply(lambda x: pd.Series.nunique(x))
# drop Sno and last update variables



df.drop('Sno', axis=1, inplace=True)

df.drop('Last Update', axis=1, inplace=True)
# convert str in datetime



df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(x).date())

# df['Last Update'] = df['Last Update'].apply(lambda x: pd.to_datetime(x).date())
# extract the day, month and year from the datetime Date object



df['Day'] = df.Date.apply(lambda x: x.day)

df['Month'] = df.Date.apply(lambda x: x.month)

df['Year'] = df.Date.apply(lambda x: x.year)
# create new variables: infected, mortality rate, recovery rate and EU



df['Infected'] = df['Confirmed'] - df['Deaths'] - df['Recovered']



df['Mortality rate'] = df['Deaths'] / df[['Infected', 'Deaths', 'Recovered']].sum(axis=1) * 100

df['Recovery rate'] = df['Recovered'] / df[['Infected', 'Deaths', 'Recovered']].sum(axis=1) * 100

df['Infected rate'] = df['Infected'] / df[['Infected', 'Deaths', 'Recovered']].sum(axis=1) * 100



# European Union Countries



EUC = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France',

'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',

'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']



# create a boolean variable that assumes 1 if the country is inside the European Union, 0 otherwhise



df['EU'] = df.Country.apply(lambda x: 1 if x in EUC else 0)
df.head()
# check missing values



def missing_values_table(dataframe):

        if dataframe.isnull().sum().sum() > 0:

            miss_val = dataframe.isnull().sum()

            miss_val_pct = 100 * miss_val / len(dataframe)

            table = pd.concat([miss_val, miss_val_pct], axis=1)

            table = table[table.loc[:, table.columns[0]] != 0].sort_values(table.columns[0], ascending=False).round(2)

            table = table.rename(columns = {0: 'Missing Values', 1: '% of Missing Values'})

            return table

        else:

            return 'there are no missing values'



missing_values_table(df)
df['Mortality rate'].fillna(0, inplace=True)

df['Recovery rate'].fillna(0, inplace=True)

df['Infected rate'].fillna(0, inplace=True)
missing_values_table(df)
# count of deaths



ax = sns.countplot(df.loc[df.Deaths.isin(df.Deaths.value_counts().nlargest(5).index), 'Deaths'], palette = 'coolwarm');

plt.title('count of deaths')



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height,

            height,

            ha='center') 
df.loc[df.Deaths == 6]
df.groupby('Date')['Confirmed'].sum().reset_index()
ax = sns.barplot(x='Date', y='Confirmed', ci=None, palette = 'coolwarm', data=df.groupby('Date')['Confirmed'].sum().reset_index())

plt.title('Confirmed cases grouped by Date')

plt.xticks(rotation=90);
sns.lineplot(x='Date', y='Mortality rate', data=df, label='Mortality rate')

sns.lineplot(x='Date', y='Recovery rate', data=df, label='Recovery rate')

# sns.lineplot(x='Date', y='Infected rate', data=df, label='Infected rate')

plt.title('Comparison between Mortality rate and Recovery rate')

plt.legend(loc='best');
plt.suptitle('Comparison between confirmed, deaths, recovered and infected patients')

plt.subplot(221)

sns.lineplot(x='Date', y='Confirmed', data=df.groupby('Date')['Confirmed'].sum().reset_index())

plt.xlabel('')

plt.xticks([])





plt.subplot(222)

ax = sns.lineplot(x='Date', y='Deaths', data=df.groupby('Date')['Deaths'].sum().reset_index(), color='red')

plt.xticks(rotation=45);

plt.xlabel('')

plt.xticks([])



plt.subplot(223)

sns.lineplot(x='Date', y='Recovered', data=df.groupby('Date')['Recovered'].sum().reset_index(), color='yellow')

plt.xticks(rotation=90)



plt.subplot(224)

sns.lineplot(x='Date', y='Infected', data=df.groupby('Date')['Infected'].sum().reset_index(), color='green')

plt.xticks(rotation=90);
df.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Infected'].sum().reset_index()
# analysis by month 



df.groupby('Month')['Confirmed'].sum().reset_index()
labels ='January', 'February'

plt.title('Confirmed cases by month')

df.groupby('Month')['Confirmed'].sum().reset_index()['Confirmed'].plot.pie(autopct=lambda x:'{:.2f}%\n({:.0f})'.\

    format(x,(x/100)*df.groupby('Month')['Confirmed'].sum().reset_index()['Confirmed'].sum()), labels=labels, 

    fontsize=10, startangle=25);
df.groupby('Month')['Deaths'].sum().reset_index()
plt.title('Deaths cases by month')

labels ='January', 'February'

df.groupby('Month')['Deaths'].sum().reset_index()['Deaths'].plot.pie(autopct=lambda x:'{:.2f}%\n({:.0f})'.\

    format(x,(x/100)*df.groupby('Month')['Deaths'].sum().reset_index()['Deaths'].sum()), labels=labels, 

    fontsize=10, startangle=25);
# analysis by Country



df.groupby('Country')['Confirmed'].max().reset_index()
ax = sns.barplot(x = 'Country', y = 'Confirmed', palette = 'coolwarm', 

            data = df.query('EU == 1').groupby('Country')['Confirmed'].max().sort_values(ascending=False).reset_index())

plt.xticks(rotation=45);



for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height,

            height,

            ha='center') 