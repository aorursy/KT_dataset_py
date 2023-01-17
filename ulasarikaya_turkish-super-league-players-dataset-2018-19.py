# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visulation tool

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.



data = pd.read_csv("../input/tsldata.csv")

data.info() #look data info

data.columns #View column names

f,ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(data.corr(),annot=True,linewidth=1,fmt=".2f",ax=ax) #

plt.show() 
data.head(3) #look data head

data.tail(4) #look data tail
plt.subplots(figsize=(15,6))

sns.set_color_codes()

sns.distplot(data['age'], color = "R")

plt.xticks(rotation=90)

plt.title('Distribution of Turkis Super League Players Age')

plt.show()

#number of turkish super league player position



plt.subplots(figsize=(15,6))

sns.countplot('position',data=data,palette='hot',edgecolor=sns.color_palette('dark',7),order=data['position'].value_counts().index)

plt.xticks(rotation=90)

plt.title('number of super league player position')

plt.show()
#most market value



dfmarketv = data.nsmallest(10, 'age').sort_values('age',ascending=True)

plt.subplots(figsize=(15,6))

sns.barplot(x="name", y="age",  data=dfmarketv ,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Top 10 youngest turkish super league player season 2018/2019')

plt.show()
data = pd.read_csv("../input/tsldata.csv")

clubs = tuple(set(data['club']))

print(clubs)
from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True)



value = []

for club in clubs:

    value.append(sum(data['market_value'].loc[data['club']==club]))

    

keys= clubs

values=value



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Market Value of players of each club")

})    
average_age = []

for club in clubs:

    average_age.append(np.mean(data['age'].loc[data['club']==club]))



keys= clubs

values=average_age



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Average Age")

})    
country, counts = np.unique(data['nationality'], return_counts=True)



keys= country

values=counts



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Nationality of Players")

})

c_value = []

for c in country:

    c_value.append(sum(data['market_value'].loc[data['nationality']==c]))



keys= country

values=c_value



iplot({

    "data": [go.Bar(x=keys, y=values)],

    "layout": go.Layout(title="Market Value vs Nationality")

})