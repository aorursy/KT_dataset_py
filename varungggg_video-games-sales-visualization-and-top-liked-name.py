import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

datas=pd.read_csv('../input/videogamesales/vgsales.csv')
datas.head(5)
datas.isnull().sum()
data= datas.dropna()

data
data.isnull().sum()
data['Year'].max(), data['Year'].min()
Platforms=data['Platform'].value_counts()

Platforms
data.groupby('Year')['Platform'].count()
data.groupby('Year', as_index=False).agg({"Platform": "sum"})
#Gloabl Sales Between 1980 to 2020

Global_Sales_date = data.groupby(['Year']).agg({'Global_Sales':['sum']})



fig, (ax1) = plt.subplots(1, figsize=(17,7))

Global_Sales_date.plot(ax=ax1)

ax1.set_title("Sales", size=13)

ax1.set_ylabel("Global sales", size=13)

ax1.set_xlabel("Year", size=13)

#Highest Sales in the Country

NA_Sales_date = data.groupby(['Year']).agg({'NA_Sales':['sum']})

EU_Sales_date = data.groupby(['Year']).agg({'EU_Sales':['sum']})

NA_EU_Sales = NA_Sales_date.join(EU_Sales_date)

JP_Sales_date = data.groupby(['Year']).agg({'JP_Sales':['sum']})

Other_Sales_date = data.groupby(['Year']).agg({'Other_Sales':['sum']})

JP_Other_Sales = JP_Sales_date.join(Other_Sales_date)







fig, (ax1) = plt.subplots(1, figsize=(17,7))

NA_EU_Sales.plot(ax=ax1)

JP_Other_Sales.plot(ax=ax1)

ax1.set_title("Sales", size=13)

ax1.set_ylabel("Y-Sales", size=13)

ax1.set_xlabel("Year", size=13)

#Top Sold Games between 1980 to 2020

Top_Sales = data.sort_values('Global_Sales', ascending = False).head(10).set_index('Name')

plt.figure(figsize=(15,10))

sns.barplot(Top_Sales['Global_Sales'], Top_Sales.index, palette='rocket')
# Top 10 liked Gaming names

plot = data.Genre.value_counts().nlargest(10).plot(kind='bar', title="Top 10 Gaming", figsize=(12,6))
# Top 10  Publisher names

plot = data.Publisher.value_counts().nlargest(10).plot(kind='bar', title="Top 10 Publisher", figsize=(12,6))
# Top 10  platform names being Played

plot = data.Platform.value_counts().nlargest(10).plot(kind='bar', title="Top 10 Platform", figsize=(12,6))
data['NA_Sales'].sum(),data['EU_Sales'].sum(),data['JP_Sales'].sum(),data['Other_Sales'].sum()
# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'

sizes = [4327, 2406, 1284, 788]

explode = (0.1, 0, 0, 0)  # only "explode" the last slice (i.e. 'Others')



fig1, ax1 = plt.subplots(figsize=(15,10))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=90)

ax1.axis('equal')  



plt.show()
data.nlargest(5, ['NA_Sales']) 
data.nlargest(5, ['EU_Sales']) 
data.nlargest(5, ['JP_Sales']) 
data.nlargest(5, ['Other_Sales']) 
data.nlargest(5, ['Global_Sales']) 
from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud( max_font_size=50, 

                       stopwords=STOPWORDS,

                       background_color='black',

                       width=600, height=300

                     ).generate(" ".join(data['Name'].sample(2000).tolist()))



plt.figure(figsize=(14,7))

plt.title("Wordcloud for Top Keywords in Names", fontsize=35)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
data.groupby('Year')['Global_Sales'].count()
data.groupby('Year')['Name'].count()
High=data.loc[(data.Year >= 2005) & (data.Year <= 2012)]

High
# Top 10 liked Gaming names between 2005 and 2012

plot = High.Name.value_counts().nlargest(10).plot(kind='bar', title="Top 10 Gaming", figsize=(12,6))
import plotly.graph_objects as go

fig = go.Figure([go.Bar(x=High.Name, y=High.Global_Sales)])

fig.show()