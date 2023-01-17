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

        

df=pd.read_csv('../input/netflix-shows/netflix_titles.csv')

import pandas as pd

import numpy as np 

from matplotlib import pyplot as plt



import seaborn as sns

sns.set_style("dark")

%matplotlib inline

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator





# Any results you write to the current directory are saved as output.
df.head()
df.isnull().sum()
# droping directior column due to large missingf director, cast, country

df.drop(df[['director', 'cast', 'country']], axis = 1, inplace = True)
# Checking for The most comman value to fill the null

most_common = df['rating']

most_common.groupby(most_common).count()
# Filling rating missisg with column avarage

df["rating"].fillna("TV-14", inplace = True) 

# Dropping the raws with missing date_added values

df=df.dropna()
# Date format

df["date_added"] = pd.to_datetime(df['date_added'])

df['day_added'] = df['date_added'].dt.day

df['year_added'] = df['date_added'].dt.year

df['month_added']=df['date_added'].dt.month

df['year_added'].astype(int);

df['day_added'].astype(int);
df.isnull().sum()


from matplotlib import rcParams

sns.countplot(df['type'], palette="Set3")

sns.set(style="darkgrid")

rcParams['figure.figsize'] = 11.7,8.27

plt.title('The difference between TV shows & Moive', fontsize = 16)

plt.xlabel('Count', fontsize = 14)

plt.ylabel('Type', fontsize = 14)


year = df[['year_added', 'show_id']].groupby('year_added').count().reset_index()

year 



sns.barplot(x="year_added", y="show_id", data=year)

sns.set(style="whitegrid")

rcParams['figure.figsize'] = 10,10

plt.title('The Highest year', fontsize = 16)

plt.xlabel('Year', fontsize = 14)

plt.ylabel('Number of new Addings', fontsize = 14)
import matplotlib.pyplot as plt

plt.style.use('seaborn-ticks')
rating = df[['rating', 'show_id']].groupby('rating').count().reset_index()

rating

labels = 'G', 'NC-17', 'NR', 'PG', 'PG-13', 'R','TV-14','TV-G','TV-MA','TV-PG','TV-Y','TV-Y7', 'TV-Y7-FV', 'UR'

sizes = rating.show_id

fig1, ax1 = plt.subplots()

patches, texts = ax1.pie(sizes, shadow=True, startangle=50, )

plt.legend(patches, labels, loc="best")

ax1.axis('equal')  



plt.show()
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.listed_in))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('category.png')

plt.show()