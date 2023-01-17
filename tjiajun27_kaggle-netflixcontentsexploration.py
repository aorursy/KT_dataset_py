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

# Read training data from csv file.
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df.head()
def data_inv(df):
    print('nums of netflix movies and shows: ',df.shape[0])
    print('nums of dataset columns: ',df.shape[1])
    print('-'*40)
    print('dateset columns: \n')
    print(df.columns)
    print('-'*40)
    print('data-type of each column: \n')
    print(df.dtypes)
    print('-'*40)
    print('missing rows in each column: \n')
    c=df.isnull().sum()
    print(c[c>0])
data_inv(df)
dups = df.duplicated(['title','country','type','release_year'])
df[dups]
df = df.drop_duplicates(['title','country','type','release_year'])
df.shape
df=df.drop('show_id', axis=1)
# Replace 'NaN' in cast coloum with with 'Unknown' 
df['cast']=df['cast'].replace(np.nan,'Unknown')
def cast_counter(cast):
    '''
    Count the number of cast.
    input arg cast: object
    return arg : int
    '''
    if cast=='Unknown':
        return 0
    else:
        lst=cast.split(', ')
        length=len(lst)
        return length
df['number_of_cast']=df['cast'].apply(cast_counter)
df=df.reset_index()
df.head()
# Replace 'NaN' in rating coloum, with mode of rating. 
df['rating']=df['rating'].fillna(df['rating'].mode()[0])
# Replace 'NaN' in date_added coloum, with '1 January' of the release_year. 
df_date_addedwNAN = df[df['date_added'].isnull()].index.tolist()
for x in (df_date_addedwNAN):
    df.loc[x, 'date_added'] = 'January 1, {}'.format(df.iloc[x]['release_year'])
    
df.head()
import re
months={
    'January':1,
    'February':2,
    'March':3,
    'April':4,
    'May':5,
    'June':6,
    'July':7,
    'August':8,
    'September':9,
    'October':10,
    'November':11,
    'December':12
}
date_lst=[]
for i in df['date_added'].values:
    datestr=re.findall('([a-zA-Z]+)\s([0-9]+)\,\s([0-9]+)',i)
    date='{}-{}-{}'.format(datestr[0][1],months[datestr[0][0]],datestr[0][2])
    date_lst.append(date)
    
df['date_added_cleaned']=date_lst
df=df.drop('date_added',axis=1)
df['date_added_cleaned']=df['date_added_cleaned'].astype('datetime64[ns]')
df['country'] = df['country'].fillna('')
df['country'] = df['country'].str.replace('\s*,\s*', ',', regex=True)
df_country = df['country'].str.split(',',expand = True)
df_country
# extracting Genres from the df
countries = dict()
for i in range(df_country.shape[1]):
    for j in range(df_country.shape[0]) :
        if (df_country[i][j] not in countries) and (df_country[i][j] != None) and df_country[i][j] != '':
            countries[df_country[i][j]] = 1
        else:
            if(df_country[i][j] != '') and df_country[i][j] != None:
                countries[df_country[i][j]] += 1
            pass  
print('Together number of countries: ',len(countries))
# Show top 6 country
{k: v for k, v in sorted(countries.items(), key=lambda item: item[1], reverse=True)[:6]}
def create_coloum(df, coloum_name):
    df[coloum_name] = [0] *df.shape[0]
    return df
    
list_countries = ['United States', 'India', 'United Kingdom', 'Canada', 'France', 'Japan', 'Others']
for country in list_countries:
    df = create_coloum(df, country)
    
for i in range(df_country.shape[0]):
    for j in range(df_country.shape[1]) :
        if(df_country[j][i]==None):
            break
        if (df_country[j][i] not in list_countries):
            df.loc[i, 'Others'] = 1
        else:
            df.loc[i, df_country[j][i]] = 1 

df = df.drop(columns=['country'])
df.head()
df['year_added']=df['date_added_cleaned'].dt.year
df.head()
df['type'].value_counts(normalize=True)
df.groupby('year_added')['type'].value_counts(normalize=True)*100
dups=df.duplicated(['title'])
df[dups]['title']
for i in df[dups]['title'].values:
    print(df[df['title']==i][['title','type','release_year']])
    print('-'*40)
plt.figure(figsize=(10,8))
df['year_added'].value_counts().plot.bar()
plt.title('Numbers of shows added each year')
plt.ylabel('Count')
plt.xlabel('year added')
plt.show()
counts=0
for i,j in zip(df['release_year'].values,df['year_added'].values):
    if i<j:
        counts+=1
print('Number of old shows added after 1 year of release: ',str(counts))
count_tvshow = []
count_movie = []
for country in list_countries:
    list_index_tvshow = df[(df['type']=='TV Show') & (df[country]==1)].index.tolist()
    count_tvshow.append(len(list_index_tvshow))
    list_index_movie = df[(df['type']=='Movie') & (df[country]==1)].index.tolist()
    count_movie.append(len(list_index_movie))
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(list_countries))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10,10))
rects1 = ax.bar(x - width/2, count_movie, width, label='movie')
rects2 = ax.bar(x + width/2, count_tvshow, width, label='tvshow')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('Count by country and type')
ax.set_xticks(x)
ax.set_xticklabels(list_countries)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()