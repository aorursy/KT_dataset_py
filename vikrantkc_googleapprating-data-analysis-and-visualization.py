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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime,date
df = pd.read_csv('/kaggle/input/googleapps-ratings/googleplaystore.csv')
df
df.shape
df.info()
df.isnull().sum()
### using matrix in msno for finding null pattern
import missingno as msno
msno.matrix(df)
msno.bar(df)
### on histogram we find the skewness in df 
df.hist()
# using median over mean because rating is rightly skewed
df.fillna(df['Rating'].median(),inplace=True)
df[df['Rating'].isnull()]
df.boxplot()
#checking for more than 5-outliers
df[df.Rating>5]
## droping outlier values
df.drop([10472],inplace=True)
df.boxplot()
# mode for categorical values
df.fillna(df['Type'].mode()[0],inplace=True)
df.fillna(df['Current Ver'].mode()[0],inplace=True)
df.fillna(df['Android Ver'].mode()[0],inplace=True)
df.isnull().sum()
### let convert price ,reviews and rating into numerical values for data analysis
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$','') if '$' in str(x) else str(x))
df['Price'] = df['Price'].apply(lambda x:float(x))
df['Reviews'] = pd.to_numeric(df['Reviews'],errors ='coerce')
## replacing in installs values and convert to float 
df['Installs'] = df['Installs'].apply(lambda x: str(x).replace('+','') if '+' in str(x) else str(x))
df['Installs'] = df['Installs'].apply(lambda x:str(x).replace(',','') if ',' in str(x) else str(x))
df['Installs'] = df['Installs'].apply(lambda x :float(x))
df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df.head(10)
plt.style.use('seaborn-white')
sns.pairplot(df)
grp = df.groupby('Category')
x = grp['Rating'].agg(np.mean)
y = grp['Price'].agg(np.sum)

z = grp['Reviews'].agg(np.mean)
plt.figure(figsize=(16,5))
plt.plot(x,'ro',color='r')
plt.xticks(rotation=90)
plt.title('Category wise rating')
plt.xlabel('Categories')
plt.ylabel('Rating')
plt.show()

plt.figure(figsize=(16,5))
plt.plot(z,'ro',color='r')
plt.xticks(rotation=90)
plt.title('Category wise rating')
plt.xlabel('Categories')
plt.ylabel('Reviews')
plt.show()

plt.figure(figsize=(16,5))
plt.plot(y,'ro',color='r')
plt.xticks(rotation=90)
plt.title('Category wise rating')
plt.xlabel('Categories')
plt.ylabel('Price')
plt.show()

plt.figure(figsize=(20,5))
fig = sns.countplot(x=df['Installs'],palette='hls')
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show()
plt.figure(figsize=(5,3))
fig = sns.countplot(x=df['Type'])
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show()
plt.figure(figsize=(10,5))
fig = sns.countplot(x=df['Content Rating'],palette='hls')
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show()
plt.figure(figsize=(30,15))
fig = sns.countplot(x = df['Category'],palette='hls')
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show()
plt.figure(figsize=(40,15))
fig = sns.countplot(x=df['Genres'],palette='hls')
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show()
plt.style.use('seaborn-white')
plt.figure(figsize=(15,10))
sns.boxplot(x='Category',y='Rating',palette='rainbow',data=df)
plt.title('Category vs Rating')
plt.xticks(rotation=90)
##mostcost app
cost = df.sort_values(by='Price',ascending=False)[['App','Price']].head(20)
cost
#top genres
df[['Genres','Rating']].groupby('Genres',as_index=False).mean().sort_values('Rating',ascending=False).head(10)
##apps with at least 200 reviews to find the best app
apps = df[df['Reviews']>=200]
apps.head()
### TOP APPS 
top_apps =  apps.sort_values(by=['Rating','Reviews','Installs'],ascending=False)
[['App','Rating','Reviews']].head(10)
top_apps
### apps with most installations
apps.sort_values(by='Installs',ascending=False)[['App','Installs','Rating']].head(15)
plt.subplots(figsize=(25,15))
wordcloud  =WordCloud(background_color='white',width=1900,height=1200).generate(" ".join(df.App))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()