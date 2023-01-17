import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", style="darkgrid", palette="deep", font="sans-serif", font_scale=1, color_codes=True)

from scipy.stats  import skew 

from scipy.stats  import kurtosis
import pandas as pd

data= pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")
data.head()
data["rate"]  = data["rate"].astype(str).apply(lambda x: x.replace('/5','') if len(x)>3 else "0")

data["rate"].unique()
data["rate"] = data["rate"].astype(float)
data["approx_cost(for two people)"] = data["approx_cost(for two people)"].astype(str).apply(lambda x: x.replace(',',''))

data["approx_cost(for two people)"].replace(["nan"],["0"],inplace=True)

data["approx_cost(for two people)"] = data["approx_cost(for two people)"].astype(int)



data["approx_cost(for two people)"].unique()
data.isnull().sum()
sns.heatmap(data.isnull(),cmap="viridis")
data = data.drop(["phone"],axis=1)

data["location"].fillna("No info",inplace =True)

data["rest_type"].fillna("No info",inplace=True)

data["cuisines"].fillna("No info",inplace=True)

data["dish_liked"].fillna("No info",inplace=True)

sns.heatmap(data.isnull(),cmap="viridis")
data.info()
data.columns
##Which are the top restaurants chains in Bengaluru

sns.countplot(y="name",data=data,order=data["name"].value_counts()[:20].index)

plt.title("Most famous restaurant chains in Bengaluru")

plt.xlabel("Number of outlets")

## How many of the restaurants do not accept online orders

sns.countplot(y="online_order",data=data)
X = data["online_order"].value_counts()

plt.pie(X,labels=X.index,startangle=90,autopct='%1.1f%%',explode = (0, 0.1))

plt.title('How many of the restaurants accept online orders')

plt.show()     

### What is the ratio b/w restaurants that provide and do not provide table booking ?    

sns.countplot(y="book_table",data=data)
X1 = data["book_table"].value_counts()

plt.pie(X1,labels=X1.index,startangle=90,autopct='%1.1f%%',explode = (0, 0.1))

plt.title('ratio b/w restaurants that provide and do not provide table booking')

plt.show() 

## Ratings distributions

sns.countplot(y="rate",data=data,order=data["rate"].value_counts().index)

plt.title("rating distribution")

sns.distplot(data["rate"])
## COST vs RATING

cost_dist = data[['rate','approx_cost(for two people)','online_order']]



sns.scatterplot(x="rate",y="approx_cost(for two people)",hue="online_order",data=cost_dist)

plt.title("cost vs rating")

plt.show()

sns.boxplot(x="online_order",y="votes",order=["Yes","No"],data=data)

## Which are the most common restaurant type in Banglore?

sns.countplot(y="rest_type",data=data,order=data["rest_type"].value_counts()[:20].index)

plt.title("most common restaurant type in Banglore")

## Cost factor

sns.countplot(y="approx_cost(for two people)",data=data,order=data["approx_cost(for two people)"].value_counts()[:20].index)

## Finding best budget restaurants

Best_budget = data[['rate','approx_cost(for two people)','location','name','rest_type']]

Best_budget .columns



def Budget(location,rest_type):

    x = Best_budget[(Best_budget['approx_cost(for two people)']<=400) & (Best_budget["rate"]>4) & (Best_budget["rest_type"]==rest_type) & (Best_budget["location"]==location)]

    return (x["name"].unique())

Budget('BTM',"Quick Bites")

## Which are the best foodie areas

sns.countplot(y="location",data=data,order=data["location"].value_counts()[:20].index)

plt.title("foodie areas")

## Which are the most common cuisines in each location

df_1 = data.groupby(['location','cuisines']).agg('count')

data_1 = df_1.sort_values(['url'],ascending=False).groupby(['location'],as_index=False).apply(lambda x : x.sort_values(by="url",ascending=False).head(3))['url'].reset_index().rename(columns={'url':'count'})

data_1.head(10)
## Which are the most popular cuisines of Bangalore?

sns.countplot(y="cuisines",data=data,order=data["cuisines"].value_counts()[:20].index)

plt.title("most popular cuisines of Bangalore")

### Wordcloud          

from wordcloud import WordCloud 

def show_wordcloud(data):

    wordcloud = WordCloud(background_color='black',max_words=200,max_font_size=40, scale=3,random_state=1 ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))

    plt.axis('off')

    plt.imshow(wordcloud,interpolation="bilinear")

    plt.show()



### A bag of words for liked dishes 

bag_list_1 = []

for i in data["dish_liked"]:

      bag_list_1.append(i.lower())

      

stop = ["no info","info","no"] 

for word in list(bag_list_1) :

    if word in stop:

         bag_list_1.remove(word)

         

import nltk      

FF = nltk.FreqDist(bag_list_1)       

FF.plot(40)
### Wordcloud for dish liked

textt = " ".join(word for word in bag_list_1)

show_wordcloud(textt)
### A bag of words for Cuisines

bag_list_2 = []

for i in data["cuisines"]:

      bag_list_2.append(i.lower())

      

### Wordcloud for cleaned  data

textt_c = " ".join(word for word in bag_list_2)

show_wordcloud(textt_c)
## Rating distribution

sns.countplot(x="rate",data=data,hue="online_order")



sns.countplot(x="rate",data=data,hue="book_table")

plt.xticks(rotation=90)
facet = sns.FacetGrid(data,hue="online_order",aspect=4)

facet.set(xlim=(2,5))

facet.map(sns.kdeplot,"rate",shade=True)

facet.add_legend()
facet = sns.FacetGrid(data,hue="book_table",aspect=4)

facet.set(xlim=(2,5))

facet.map(sns.kdeplot,"rate",shade=True)

facet.add_legend()
