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

%matplotlib inline

df = pd.read_csv("../input/videogamesales/vgsales.csv")

df.head()
df.shape
df.info()
#Null values

df.isna().sum()
df['Publisher'].unique()
df['Publisher'].value_counts()[:10]
df.dropna(how="any",inplace=True)

df.isna().sum()
df.shape
df.columns
df.Genre.value_counts()
plt.rcParams['figure.figsize'] = (12,9)

sns.countplot(x=df['Genre'],data = df,palette= "Reds_r")

plt.title("Number of Games od Different Genres")

plt.xlabel("Genre")

plt.ylabel("Count")

plt.show()
plt.rcParams['figure.figsize'] = (12,9)

sns.barplot(x=df['Genre'],y=df['Global_Sales'],data = df.sort_values('Global_Sales',ascending=False))

plt.title("Global sales of Differet Genre Games")

plt.xlabel("Genre")

plt.ylabel("Global Sales ")

plt.show()
max_comp = df['Publisher'].value_counts()[:20]
plt.rcParams['figure.figsize'] = (15,9)

max_comp.plot(kind="bar",color='#e35e43')

plt.title("Top 20 companies with most number of Games")

plt.xlabel("Companies")

plt.ylabel("Counts")

plt.show()
#top 20 company sales

plt.rcParams['figure.figsize'] = (15,9)

def zone_based_sales(publisher):

    sales = df.groupby([publisher])['Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales'].sum().sort_values('Global_Sales',ascending=False)[:20]

    sales_ = pd.DataFrame(sales)

    sales_.plot(kind='bar')

    plt.title("Zone Based Sales")

    plt.xlabel("Companies")

    plt.ylabel("Sales")

    plt.show()

    

    

    

    
zone_based_sales('Publisher')
zone_based_sales('Genre')
zone_based_sales('Platform')
plt.rcParams['figure.figsize'] = (12,9)

sns.countplot(x=df['Year'],data = df,palette="Reds_r")

plt.title("Frequency of Game release")

plt.xlabel("Years")

plt.ylabel("Frequency")

plt.xticks(rotation=90)

plt.show()
plt.rcParams['figure.figsize'] = (30,10)

sns.countplot(x=df['Year'],data=df,hue='Genre',order=df.Year.value_counts().iloc[:5].index)

plt.xticks(size=16,rotation=90)

plt.show()
data_year = df.groupby(['Year'])['Global_Sales'].sum()

data_year = data_year.reset_index()
plt.rcParams['figure.figsize'] = (12,9)

sns.barplot(x='Year',y='Global_Sales',data=data_year)

plt.xticks(rotation=90)

plt.show()
game_publish = df.groupby(['Publisher'])['Platform'].count()

game_publish = game_publish.reset_index()
plt.rcParams['figure.figsize'] = (20,10)

sns.barplot(x='Publisher',y='Platform',data=game_publish.sort_values('Platform',ascending=False)[:10])

plt.show()
games = df[['Name','Year','Genre','Global_Sales']]

top_games  = games.sort_values('Global_Sales',ascending=False)[:20]

name = top_games['Name']

year = top_games['Year']

y = np.arange(0,20)

plt.rcParams['figure.figsize'] = (12,9)

g = sns.barplot(x='Name',y='Global_Sales',data = top_games)

index = 0

for value in top_games['Global_Sales']:

    g.text(index,value-18,name[index],color="#001",size=14,rotation=90,ha='center')

    index += 1

    

plt.xticks(y,top_games['Year'],fontsize=14,rotation=90)

plt.xlabel("Release Year")

plt.show()