# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
df.head(10)
df.shape
df.describe()
df.boxplot()
df.hist()
df.info()
df.isnull().sum()
df[df.Rating>5]
df.drop([10472],inplace=True)
df.boxplot()
df.hist()
thresh=len(df)*0.1

thresh
df.dropna(thresh=thresh,axis=1,inplace=True)
df.isnull().sum()
def impute_median(series):

    return series.fillna(series.median())
df.Rating=df['Rating'].transform(impute_median)
df.isnull().sum()
df['Type'].fillna(str(df['Type'].mode().values[0]),inplace=True)

df['Current Ver'].fillna(str(df['Current Ver'].mode().values[0]),inplace=True)

df['Android Ver'].fillna(str(df['Android Ver'].mode().values[0]),inplace=True)
df.isnull().sum()
df['Price']=df['Price'].apply(lambda x:float(str(x).replace('$','')))

df['Reviews']=pd.to_numeric(df['Reviews'],errors='coerce')
df['Installs'] = df['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else str(x))

df['Installs'] = df['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))

df['Installs'] = df['Installs'].apply(lambda x: float(x))
df.head(10)
df.describe()
df.shape
plt.figure(figsize=(12,9))

sns.kdeplot(df.Rating,color='Red',shade=True)

plt.xlabel('Rating')

plt.ylabel('Frequency')

plt.title('Distribution of Rating')
df['Category'].describe()
df['Category'].unique()
plt.figure(figsize=(12,10))

sns.countplot(x='Category',data=df)

plt.xticks(rotation=80)

plt.ylabel('Count')

plt.title('Count of apps in each category')
plt.figure(figsize=(12,10))

sns.boxplot(x='Category',y='Rating',data=df)

plt.xticks(rotation=90)

plt.ylabel('Rating')

plt.title('Boxplot of Rating vs Category')
plt.figure(figsize=(12,9))

sns.kdeplot(df.Reviews,color='Red',shade=True)

plt.xlabel('Reviews')

plt.ylabel('Frequency')

plt.title('Distribution of Reviews')
df[df.Reviews>500000]
plt.figure(figsize=(10,10))

sns.jointplot(x='Reviews',y='Rating',data=df,color='green',size=9)
plt.figure(figsize=(12,12))

sns.regplot(x='Reviews',y='Rating',data=df[df.Reviews<1000000],color='orange')

plt.title('Rating vs Reviews')
df['Size'].replace('Varies with device', np.nan, inplace = True )
df.Size = (df.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \

             df.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)

            .fillna(1)

            .replace(['k','M'], [10**3, 10**6]).astype(int))
df['Size'].fillna(df.groupby('Category')['Size'].transform('mean'),inplace = True)

plt.figure(figsize=(10,10))

sns.jointplot(x='Size',y='Rating',data=df,color='green',size=10)
Sorted_value = sorted(list(df['Installs'].unique()))

df['Installs'].replace(Sorted_value,range(0,len(Sorted_value),1), inplace = True )
df.Installs

plt.figure(figsize = (12,10))

sns.regplot(x="Installs", y="Rating", color = 'teal',data=df);

plt.title('Rating VS Installs')
plt.figure(figsize=(10,10))

plt.pie(x=df['Type'].value_counts(),labels=df['Type'].value_counts().index,explode=(0.1,0.1),autopct='%1.1f%%', shadow=True, startangle=270)

plt.title('Percentage of free app in store',size=15)
df['Free'] = df['Type'].map(lambda s :1  if s =='Free' else 0)

df.drop(['Type'], axis=1, inplace=True)
df.Free.head()
plt.figure(figsize = (10,10))

sns.regplot(x="Price", y="Rating", color = 'teal',data=df[df['Reviews']<1000000]);

plt.title('Scatter plot Rating VS Price',size = 15)
df.loc[ df['Price'] == 0, 'PriceBand'] = '0 Free'

df.loc[(df['Price'] > 0) & (df['Price'] <= 0.99), 'PriceBand'] = '1 cheap'

df.loc[(df['Price'] > 0.99) & (df['Price'] <= 2.99), 'PriceBand']   = '2 not cheap'

df.loc[(df['Price'] > 2.99) & (df['Price'] <= 4.99), 'PriceBand']   = '3 normal'

df.loc[(df['Price'] > 4.99) & (df['Price'] <= 14.99), 'PriceBand']   = '4 expensive'

df.loc[(df['Price'] > 14.99) & (df['Price'] <= 29.99), 'PriceBand']   = '5 too expensive'

df.loc[(df['Price'] > 29.99), 'PriceBand']  = '6 Very Very expensive'
df[['PriceBand', 'Rating']].groupby(['PriceBand'], as_index=False).mean()

sns.catplot(x="PriceBand",y="Rating",data=df, kind="boxen",size=10)

plt.xticks(rotation=90)

plt.ylabel("Rating")

plt.title('Boxen plot Rating VS PriceBand',size = 20)
sns.catplot(x="Content Rating",y="Rating",data=df, kind="box", height = 10 )

plt.xticks(rotation=90)

plt.ylabel("Rating")

plt.title('Box plot Rating VS Content Rating',size = 20)
df = df[df['Content Rating'] != 'Unrated']

df = pd.get_dummies(df, columns= ["Content Rating"])

df['Genres'].describe()
df.Genres.unique()
df.Genres.value_counts()
df['Genres']=df['Genres'].str.split(';').str[0]
df.Genres.describe()
df.Genres.unique()
df.Genres.value_counts().tail()
df['Genres'].replace('Music & Audio', 'Music',inplace = True)

df[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().describe()

df[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating',ascending=False).head(1)

df[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating',ascending=False).tail(1)

sns.catplot(x='Genres',y='Rating',data=df,kind='boxen',height=11)

plt.xticks(rotation=90)

plt.ylabel('Rating')

plt.title('Boxplot of Rating vs Genres')