# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
zomato = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')
zomato.head(5)
#removing unnecsary features such as url,address and phone

del zomato['url']

del zomato['phone']

del zomato['address']
zomato['location'].value_counts().head(20).plot.bar()
zomato['location'].value_counts().tail(20).plot.bar()
zomato['rate'].value_counts().sort_index().plot.line()
zomato['rate'] = zomato['rate'].str.extract('(\d\.\d)', expand=True)
zomato['rate'] = zomato['rate'].astype(float)
zomato['approx_cost(for two people)'].unique()
#Let's clean it

zomato['approx_cost(for two people)'] = zomato['approx_cost(for two people)'].str.replace(',','')
zomato['approx_cost(for two people)'].fillna(0)
zomato['rate'].value_counts().sort_index().plot.bar()
zomato.votes.describe()
zomato['votes'].plot.hist()
def normalize(column):

    upper = column.max()

    lower = column.min()

    y = (column - lower)/(upper-lower)

    return y
helpful_votes = normalize(zomato.votes)

helpful_votes.describe()
helpful_votes.plot.hist()
def sigmoid(x):

    e = np.exp(1)

    y = 1/(1+e**(-x))

    return y
sigmoid_votes = sigmoid(zomato.votes)

sigmoid_votes.describe()
sigmoid_votes.plot.hist()
helpful_log = np.log(zomato.votes+1)
helpful_log.describe()
helpful_log.plot.hist()
helpful_log_normalized = normalize(helpful_log)

helpful_log_normalized.describe()
helpful_log_normalized.plot.hist()
cost_dist=zomato[['rate','approx_cost(for two people)','online_order']].dropna()

cost_dist['approx_cost(for two people)']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
plt.figure(figsize=(10,7))

sns.scatterplot(x="rate",y='approx_cost(for two people)',hue='online_order',data=cost_dist)

plt.show()
plt.figure(figsize=(6,6))

sns.distplot(cost_dist['approx_cost(for two people)'])

plt.show()
votes_yes=zomato[zomato['online_order']=="Yes"]['votes']

trace0=go.Box(y=votes_yes,name="accepting online orders",

              marker = dict(

        color = 'rgb(214, 12, 140)',

    ))



votes_no=zomato[zomato['online_order']=="No"]['votes']

trace1=go.Box(y=votes_no,name="Not accepting online orders",

              marker = dict(

        color = 'rgb(0, 128, 128)',

    ))



layout = go.Layout(

    title = "Box Plots of votes",width=800,height=500

)



data=[trace0,trace1]

fig=go.Figure(data=data,layout=layout)

py.iplot(fig)
plt.figure(figsize=(7,7))

cuisines=zomato['dish_liked'].value_counts()[:10]

sns.barplot(cuisines,cuisines.index)

plt.xlabel('Count')

plt.title("Most liked dishes in Bangalore")
biryani_=zomato[zomato['dish_liked'].isin(['Biryani', 'Chicken Biryani']) ][zomato['online_order']=="Yes"]
biryani_['approx_cost(for two people)'] = biryani_['approx_cost(for two people)'].astype(int)
plt.figure(figsize=(20,20))

biryani_.reset_index().plot.scatter(x = 'rate', y = 'approx_cost(for two people)')

plt.show()
plt.figure(figsize=(10,7))

sns.scatterplot(x="rate",y='rest_type',data=biryani_)

plt.show()
biryani_r=biryani_[biryani_['rate']>=4]
plt.figure(figsize=(10,7))

sns.scatterplot(x="rate",y='location',data=biryani_r)

plt.show()
plt.figure(figsize=(7,7))

cuisines=zomato['cuisines'].value_counts()[:10]

sns.barplot(cuisines,cuisines.index)

plt.xlabel('Count')

plt.title("Most popular cuisines of Bangalore")
cuisine_=zomato[zomato['cuisines'].isin(['North Indian', 'North Indian, Chinese','South Indian','Biryani','South Indian, North Indian, Chinese']) ][zomato['rate']>=4]
plt.figure(figsize=(7,7))

loc=cuisine_['location'].value_counts()[:10]

sns.barplot(loc,loc.index)

plt.xlabel('Count')

plt.title("Most popular locations serving Noth Indian,Chinese,South Indian and Biryani")