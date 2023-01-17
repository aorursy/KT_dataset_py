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
r=pd.read_csv('/kaggle/input/windows-store/msft.csv',parse_dates=['Date'])

r.head()
r[['people_ratings']]=r[['No of people Rated']]

del r['No of people Rated']
r.head()
r.info()
r.isnull().sum()
#identifying the missing row

d=r[r['Name'].isnull()]

d
#dropping missing row

r.dropna(inplace=True)
r.isnull().sum()
r.tail()
r['Rating'].value_counts()/len(r)
r['Category'].value_counts()/len(r)
#denoting non free values as paid

r.loc[r['Price'] != 'Free', 'Price'] = 'Paid' 
r['Price'].value_counts()/len(r)*100
#finding highest rated apps in each category

r[r['Rating']>=1.0].groupby(['Category']).max()
#mean rating and no of people rated in each category

r.groupby('Category')[['Rating','people_ratings']].mean()
import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly

plotly.offline.init_notebook_mode(connected = True)
sns.barplot('Price','Rating',data=r)

plt.title("Distribution of Apps")
#finding majority of paid and free apps in each category

plt.figure(figsize=(20,10))

fig=px.scatter(r,'Category',color='Price')

fig.show()
#displaying proporations of total ratings of people of each category

fig=px.pie(r,values='people_ratings',names='Category',title="Distribution of Categories according to user's ratings")

fig.show()
#finding rating trend of apps given by users in each category

plt.figure(figsize=(30,20))

sns.lineplot('Category','people_ratings',data=r,hue='Rating',marker='o',legend=False)

plt.xticks(rotation=90)

plt.legend(title='Ratings',loc='upper right',labels=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])

plt.title('Distribution of Categories based on rating count given by users',fontsize=13)
fig = px.scatter(r, x="Rating", y="people_ratings", 

                 color="Category", 

                 hover_data=['people_ratings','Rating','Category','Price'], 

                 title = "Visualization of each app features of Window Store")

fig.show()
#setting date as index for time series analysis

r=r.set_index('Date')
r['Month']=r.index.month

r['Year']=r.index.year

r['Day']=r.index.day
#determining most number of downloads per year

fig=px.pie(r,values='people_ratings',names='Year',title='Number of ratings on yearly basis')

fig.show()
fig=px.pie(r,values='people_ratings',names='Month',title='Number of ratings on monthly basis')

fig.show()
fig=px.pie(r,values='people_ratings',names='Day',title='Number of ratings on daily basis')

fig.show()