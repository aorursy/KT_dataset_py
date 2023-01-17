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

import seaborn as sns
df=pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')

df.head()
df.Year.unique()
df=df.dropna()

df.info()

#查看发现数据集存在相对少量缺少，直接drop
df.Year=df.Year.astype(int)

#df.Year=pd.to_datetime(df.Year.astype(str),format='%Y')
df.info()
#去除2016年以后的数据

df=df[df.Year<=2016]
#按年份排序

df=df.sort_values(by = 'Year')
df.reset_index(level=None, drop=True, inplace=True, col_level=0) 

df.head()
%matplotlib inline
colorp=['lavender','mediumspringgreen','mediumaquamarine','aquamarine',

        'turquoise','lightseagreen','mediumturquoise','lightblue',

        'lightcyan','paleturquoise','darkslategray','teal']

#备选颜色方案
platform = df.Platform.unique()

print('Including this platform:')

print(platform)

genre = df.Genre.unique()

print('')

print('Including this genre:')

print(genre)
fig, (ax1, ax2,ax3) = plt.subplots(1,3, figsize = (21,5))



ax1.plot(df.groupby('Year').Global_Sales.sum().index,

         df.groupby('Year').Global_Sales.sum());

ax1.set_title('Global Sales')



ax2.plot(df.groupby('Year').Name.count().index,

         df.groupby('Year').Name.count());

ax2.set_title('Number of Games')



ax3.plot(df.groupby(['Year','Genre']).agg({'Name':'count'}).unstack().notnull().sum(axis=1).index,

         df.groupby(['Year','Genre']).agg({'Name':'count'}).unstack().notnull().sum(axis=1).values)

ax3.set_title('Genre Development');

#销量前10的平台

yp=df.groupby('Platform').Global_Sales.sum().sort_values(ascending=False).index[0:10]

yp
plt.figure(figsize=(20,5))

for p in range(len(yp)):

    plt.bar(df[df['Platform']==yp[p]].groupby('Year').Global_Sales.sum().index,

            df[df['Platform']==yp[p]].groupby('Year').Global_Sales.sum().values,label=yp[p],color=colorp[p+2])

plt.title('Sales of Main Platforms')

plt.legend();
plt.figure(figsize=(18,12))

for g in genre:

    plt.barh(df[df['Genre']==g].groupby('Year').Global_Sales.sum().index,

             df[df['Genre']==g].groupby('Year').Global_Sales.sum().values,label=g)

plt.title('Sales of Different Genre')

plt.legend();
#我们试图深入1.1中的第二个猜想（2000年和2005年出现波动是否和游戏平台的迭代有关），查看各个游戏平台第一次出现的时间

for p in platform:

    print(p,df[df['Platform']==p].Year.iloc[0])
for g in genre:

    print(g,df[df['Genre']==g].Year.iloc[0])
plt.figure(figsize=(20,5))

for a in ['NA_Sales','EU_Sales','JP_Sales','Other_Sales']:

    plt.plot(df.groupby('Year')[a].sum().index,

             df.groupby('Year')[a].sum().values,label=a)

plt.title('Area Sales')

plt.legend();
fig, (ax1,ax2) = plt.subplots(2,1, figsize = (21,10))

ax1.barh(df.groupby('Genre')['NA_Sales'].sum().index,

        df.groupby('Genre')['NA_Sales'].sum().values,label='NA_Sales',facecolor='y')



ax1.barh(df.groupby('Genre')['EU_Sales'].sum().index,

        -df.groupby('Genre')['EU_Sales'].sum().values,label='EU_Sales',facecolor='gold')

ax1.set_xticks([-600,-400,-200,0,200,400,600,800,1000])

ax1.set_xticklabels([600,400,200,0,200,400,600,800,1000])

ax1.legend()



ax2.barh(df.groupby('Genre')['JP_Sales'].sum().index,

        df.groupby('Genre')['JP_Sales'].sum().values,label='JP_Sales',facecolor='cyan')



ax2.barh(df.groupby('Genre')['Other_Sales'].sum().index,

        -df.groupby('Genre')['Other_Sales'].sum().values,label='Other_Sales',facecolor='c')

ax2.set_xticks([-600,-400,-200,0,200,400,600,800,1000])

ax2.set_xticklabels([600,400,200,0,200,400,600,800,1000])

ax2.legend();
plt.figure(figsize=(18,10))

plt.barh(df.groupby('Publisher').Global_Sales.sum().sort_values(ascending=False)[0:20].index,

         df.groupby('Publisher').Global_Sales.sum().sort_values(ascending=False)[0:20].values);
top_10_company=df.groupby('Publisher').Global_Sales.sum().sort_values(ascending=False)[0:10].index.to_list()

print('top 10 Publisher:''\n',top_10_company)
plt.figure(figsize=(20,5))

for c in top_10_company[0:5]:

    x=((df[df['Publisher']==c].groupby('Year').Global_Sales.sum())/(df.groupby('Year').Global_Sales.sum()))

    plt.bar(x.index,x.values,label=c)

plt.title('Market Share of top5 Publisher')

plt.legend();
for c in top_10_company[0:7]:

    plt.figure(figsize=(20,5))

    plt.bar(df[df['Publisher']==c].groupby('Year').NA_Sales.sum().index,

             df[df['Publisher']==c].groupby('Year').NA_Sales.sum().values,facecolor='y',label='NA')

    

    plt.bar(df[df['Publisher']==c].groupby('Year').EU_Sales.sum().index,

             df[df['Publisher']==c].groupby('Year').EU_Sales.sum().values,facecolor='gold',label='EU')

    

    plt.bar(df[df['Publisher']==c].groupby('Year').JP_Sales.sum().index,

             df[df['Publisher']==c].groupby('Year').JP_Sales.sum().values,facecolor='cyan',label='JP')

    

    plt.bar(df[df['Publisher']==c].groupby('Year').Other_Sales.sum().index,

             df[df['Publisher']==c].groupby('Year').Other_Sales.sum().values,facecolor='c',label='Others')

    plt.xticks(np.linspace(1980,2016,5))

    plt.title(c)

    plt.legend()

for c in top_10_company[0:7]:

    plt.figure(figsize=(20,5))

    plt.bar(df[df['Publisher']==c].groupby('Platform').Name.count().index,

            df[df['Publisher']==c].groupby('Platform').Name.count().values,facecolor='mediumturquoise')

    plt.bar(df[df['Publisher']==c].groupby('Platform').Global_Sales.sum().index,

            -df[df['Publisher']==c].groupby('Platform').Global_Sales.sum().values,facecolor='paleturquoise')

    plt.yticks([-400,-300,-200,-100,0,100,200],

              ['400m$','300m$','200m$','100m$',0,100,200])

    plt.title('GameNumber vs Sales of '+c+' in different platforms')

    

topgame=df[df['Rank']<=30].sort_values(by='Rank')[['Rank','Year','Name','Platform','Genre','Publisher','Global_Sales']]

topgame

#历史销量前30的游戏
print(topgame.Year.value_counts())

print('')

print(topgame.Genre.value_counts())

print('')

print(topgame.Publisher.value_counts())
ys = df.groupby(['Year']).Global_Sales.max().values

ygame = df[df['Global_Sales'].isin(ys)].iloc[:,0:6]

ygame
print(ygame.Genre.value_counts())

print('')

print(ygame.Publisher.value_counts())