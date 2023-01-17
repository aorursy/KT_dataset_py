# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/attacks.csv', encoding = 'ISO-8859-1')
df.head()
dates = pd.to_datetime(df['Date'],dayfirst=True, errors='coerce')
year = dates.dropna().map(lambda x:x.year)

year_counter = Counter(year).most_common(10)

year_index = [year[0] for year in year_counter]

year_values = [year[1] for year in year_counter]



fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = year_index,y=year_values,ax=ax)

plt.title('Top Ten Year')

plt.xlabel('Year')

plt.ylabel('Counter')
table_count = df['Type'].value_counts()

type_index = table_count.index

type_values = table_count.values



fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = type_values , y = type_index , ax =ax , orient='h')

plt.title('Total Type')

plt.xlabel('Counter')

plt.ylabel('Type')
df_copy = df.copy()

df_copy['Year'] = pd.to_datetime(df_copy['Date'],dayfirst=True, errors='coerce')

df_copy['Year'].dropna(axis=0,inplace=True)

table_count = pd.pivot_table(data = df_copy[df_copy['Country'] == 'AUSTRALIA'],

                             index = ['Type'],

                             columns=['Area'],

                             values = ['Year'],

                             aggfunc = 'count')

fig,ax = plt.subplots(figsize=(8,6))

sns.heatmap(table_count['Year'],ax=ax,annot=True,fmt='2.0f',vmin=0,linewidth=.5)

plt.title('AUSTRALIA vs Type')
table_count = df.groupby(df['Country'])['Type'].size()

table_count = table_count.sort_values(ascending=False)[:20]

table_count_index = table_count.index

table_count_values = table_count.values



table_count = pd.pivot_table(data = df_copy[df_copy['Country'].isin(table_count_index)],

               index = ['Type'],columns=['Country'],values=['Year'],aggfunc='count')

fig,ax = plt.subplots(figsize=(8,6))

sns.heatmap(table_count['Year'],ax=ax,vmin=0,linewidth=.5,annot=False)
table_count = Counter(df_copy['Activity'].dropna().tolist()).most_common(20)

table_count_index = [table[0] for table in table_count]

table_count_values = [table[1] for table in table_count]

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = table_count_values , y=table_count_index,ax=ax,orient='h')

plt.title('Top 20 Activity')

plt.xlabel('Counter')

plt.ylabel('Year')
df_copy2 = df_copy.copy()

df_copy2['Year'] = df_copy['Year'].map(lambda x:x.year)

year_range = np.arange(2010,2017).tolist()

fig,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(data =df_copy2[df_copy2['Year'].isin(year_range)],x='Year',hue='Type',ax=ax)

plt.legend(loc='best',bbox_to_anchor=(1.05, 1))

plt.title('Type vs Year(2010-2016)')
def age_process(age):

    try:

        age = int(age)

    except:

        age = 0

    if (age > 0 and age <= 100):

        return age

    else:

        return np.nan

df['Age'].fillna(0,inplace=True)

df['Age'] = df['Age'].apply(age_process)

fig,ax = plt.subplots(figsize=(8,6))

ax = sns.distplot(df['Age'].dropna().astype(np.int32),

             ax=ax,

             hist_kws={"alpha": 0.6, "color": "#ffcc80"},

             kde=False,bins=15)

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Age Distribution')
country_list = df['Country'].dropna().unique()

for country in country_list:

    curr_country_counter = df[df['Country'] == country]['Area'].dropna().value_counts()

    curr_country_index = curr_country_counter.index

    if len(curr_country_index) <=3:

        continue

    curr_country_values = curr_country_counter.values

    fig,ax = plt.subplots(figsize=(8,6))

    sns.barplot(x = curr_country_index,y=curr_country_values,ax=ax)

    if len(curr_country_index) >=5:

        ticks = plt.setp(ax.get_xticklabels(),rotation=90)

    plt.title('%s' %(country))