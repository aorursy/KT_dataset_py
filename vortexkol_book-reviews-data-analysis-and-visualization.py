# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Users

u_cols = ['user_id', 'location', 'age']

users = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Users.csv', sep=';', names=u_cols, encoding='latin-1',low_memory=False)



#Books

i_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']

items = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Books.csv', sep=';', names=i_cols, encoding='latin-1',low_memory=False)



#Ratings

r_cols = ['user_id', 'isbn', 'rating']

ratings = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1',low_memory=False)
users.head(10)
items.head()
ratings.head()
users.drop(users.index[0],inplace=True)

items.drop(items.index[0],inplace=True)

ratings.drop(ratings.index[0],inplace=True)
users.head()
items.head()
ratings.head()
df=pd.merge(users,ratings,on='user_id')

df=pd.merge(items,df,on='isbn')

df.head()
df.shape
df.describe()
df.info()
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.info()
df.drop(['isbn','img_s','img_m','img_l','user_id'],axis=1,inplace=True)

df.head()
df['age']=df['age'].astype(int)
location = df.location.str.split(', ', n=2, expand=True)



df['city'] = location[0]

df['state'] = location[1]

df['country'] = location[2]
df.head()
df.drop('location',axis=1,inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
df.shape
plt.figure(figsize=(12,8))

sns.countplot(x='rating',data=df)

plt.title('Rating Distribution',size=20)
ds=df['year_of_publication'].value_counts().head(50).reset_index()

ds.columns=['year','count']

ds.head()

ds['year']=ds['year']+'year'
plt.figure(figsize=(12,10))

sns.barplot(x='count',y='year',data=ds)

plt.ylabel('Year Of Publication')

plt.title('Years of publishing',size=20)
ds=df['book_author'].value_counts().head(50).reset_index()

ds.columns=['author','count']

ds.head()
plt.figure(figsize=(12,12))

sns.barplot(x='count',y='author',data=ds)

plt.xlabel('Author')

plt.ylabel('Count')

plt.title('Authors with most Ratings',size=20)
ds=df['book_title'].value_counts().head(50).reset_index()

ds.columns=['book','count']

ds.head()
plt.figure(figsize=(12,12))

sns.barplot(x='count',y='book',data=ds)

plt.xlabel('Book')

plt.ylabel('Count')

plt.title('Books with most Ratings',size=20)
ds=df['country'].value_counts().head(50).reset_index()

ds.columns=['country','count']

ds.head(20)
plt.figure(figsize=(12,12))

sns.barplot(x='count',y='country',data=ds.head(15))

plt.xlabel('Count')

plt.ylabel('Country')

plt.title('Countries with most Ratings',size=20)
plt.figure(figsize=(12,8))

sns.distplot(df['age'],kde=False)

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.title('Age Distribution',size=20)
ds=df.groupby('rating')['age'].mean().reset_index()

ds.info()
ds['rating']=ds['rating'].astype(int)
plt.figure(figsize=(12,8))

sns.barplot(x='rating',y='age',data=ds)

plt.xlabel('Ratings')

plt.ylabel('Average Age')

plt.title('Average Age for every Rating',size=20)
from wordcloud import WordCloud,STOPWORDS

stop_words=set(STOPWORDS)





author_string = " ".join(df['book_author'].astype(str))

title_string = " ".join(df['book_title'].astype(str))

publisher_string = " ".join(df['publisher'].astype(str))
wc = WordCloud(width=800,height=500, max_font_size=100,stopwords=stop_words,background_color='white').generate(author_string)

fig=plt.figure(figsize=(16,8))

plt.axis('off')

plt.title('Wordcloud of Famous Authors',size=20)

plt.imshow(wc)
wc = WordCloud(width=800,height=500, max_font_size=100,stopwords=stop_words,background_color='white').generate(title_string)

fig=plt.figure(figsize=(16,8))

plt.axis('off')

plt.title('Wordcloud of Most Rated titles',size=20)

plt.imshow(wc)
wc = WordCloud(width=800,height=500, max_font_size=100,stopwords=stop_words,background_color='white').generate(publisher_string)

fig=plt.figure(figsize=(16,8))

plt.axis('off')

plt.title('Wordcloud of Famous Publishers',size=20)

plt.imshow(wc)