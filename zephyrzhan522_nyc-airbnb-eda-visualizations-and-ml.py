from __future__ import division

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





import random



from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
#Loading data

airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb.head()
#drop columns not used

airbnb =airbnb.drop(['reviews_per_month','host_id'], axis=1)
#check if there are null values

airbnb.isnull().sum()
#There are only few null values, lets just fill them with something

airbnb.fillna(value='missing values',inplace=True)
#Checking statics

airbnb.describe()
groupby_night=airbnb.groupby(['minimum_nights']).count().reset_index()

plt.plot(groupby_night.minimum_nights,groupby_night.id)

plt.xlabel('Minimum Nights')

plt.ylabel('Count')
#Try minimum nights smaller than 50

groupby_night= groupby_night[groupby_night.minimum_nights<50]



plt.plot(groupby_night.minimum_nights,groupby_night.id)

plt.xlabel('Minimum Nights')

plt.ylabel('Count')
bighost = airbnb.groupby(["host_name","neighbourhood_group"]).agg({'calculated_host_listings_count': 'sum'}).sort_values('calculated_host_listings_count',ascending=False)

bighost.reset_index(inplace=True)

bighost.head(10).plot(kind="bar",x='host_name',figsize =(12,8))

plt.xticks(rotation=-45)
#Most popular keywords for hotel name

from wordcloud import WordCloud, ImageColorGenerator

from wordcloud import WordCloud, STOPWORDS



text = "".join(str(each) for each in airbnb.name)



def red_color_func(word, font_size, position, orientation, random_state=None,**kwargs):

    return "hsl(50, 100%%, %d%%)" % random.randint(60, 100)



# adding movie script specific stopwords

stopwords = set(STOPWORDS)

stopwords.add("int")

stopwords.add("ext")



wc = WordCloud(max_words=10000,stopwords=stopwords, margin=10,

               random_state=3).generate(text)



# Display the generated image:

plt.figure(figsize=(10,8))

plt.imshow(wc.recolor(color_func=red_color_func, random_state=1),

           interpolation="bilinear")

plt.axis("off")
#make room_type into integer data by using mapping

airbnb['room_type2'] = airbnb['room_type']

rtype = {'Private room':1, 'Entire home/apt':2, 'Shared room':3}

airbnb['room_type2'] = airbnb['room_type2'].map(lambda x : rtype[x])

#Now we know room type is most relevant variable to price so far.
#neighbourhood_group, make it into int by using mapping

airbnb['neighbourhood_group2'] = airbnb['neighbourhood_group']

airbnb['neighbourhood_group2'] = airbnb['neighbourhood_group2'].replace(['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx'],[1,2,3,4,5])
#correlation

corr =airbnb.corr(method='kendall')

#use kendall relevant coeffcient

plt.figure(figsize=(12,10))

sns.heatmap(corr, annot=True)

plt.xticks(rotation=-45)
#group by room type

f, ax = plt.subplots(figsize=(10, 6))

#ax.set_xscale("log")

sns.boxplot(data=airbnb[airbnb['price']<300],y='room_type',x='price',width=.6,palette='plasma')

#sns.stripplot(data=airbnb[airbnb['price']<300],y="room_type", x="price", size=2, color=".4", linewidth=0)



ax.xaxis.grid(True)

ax.set(ylabel="")
# neighbourhood_group

f, ax = plt.subplots(figsize=(10, 6))



sns.boxplot(data=airbnb[airbnb['price']<300],y='neighbourhood_group',x='price',width=.6,palette='plasma')

#sns.stripplot(data=airbnb[airbnb['price']<300],y="neighbourhood_group", x="price", size=2, color=".4", linewidth=0)



ax.xaxis.grid(True)

ax.set(ylabel="")
# neighbourhood

f, ax = plt.subplots(figsize=(10, 12))



sns.boxplot(data=airbnb[(airbnb['price']<300) & (airbnb['neighbourhood_group']=='Brooklyn')],y='neighbourhood',x='price',width=.6,palette='plasma')

#sns.stripplot(data=airbnb[airbnb['price']<300],y="neighbourhood_group", x="price", size=2, color=".4", linewidth=0)



ax.xaxis.grid(True)

ax.set(ylabel="")
#KNN

import sklearn

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.neighbors import KNeighborsClassifier

room1 =  airbnb[(airbnb['price']<175) & (airbnb['room_type']=='Private room')]

X = room1[['latitude','longitude']].values

y = (room1['price']/10).astype(int)

y = y.values



#split train/test set

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=1/3,random_state=3)



rmse = []

from sklearn.metrics import mean_squared_error

k_range = range(1,21)



for k in k_range:

    #choose best K value

    best_knn = KNeighborsClassifier(n_neighbors=k)

    #train model

    best_knn.fit(train_X,train_y)

    #check score

    print(best_knn.score(test_X,test_y))

    

    predict_y = best_knn.predict(test_X)

    

    #RMSE

    rmse.append(mean_squared_error(test_y,predict_y)**0.5)

    

plt.plot(k_range,rmse)

plt.xlabel('Value of K for KNN')

plt.ylabel('Error')
room2 =  airbnb[(airbnb['price']<175) & (airbnb['room_type']=='Shared room')]

X = room2[['latitude','longitude']].values

y = (room2['price']/10).astype(int)

y = y.values



#split train/test set

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=1/3,random_state=3)



rmse = []

from sklearn.metrics import mean_squared_error

k_range = range(1,21)



for k in k_range:

    #choose best K value

    best_knn = KNeighborsClassifier(n_neighbors=k)

    #train model

    best_knn.fit(train_X,train_y)

    #check score

    print(best_knn.score(test_X,test_y))

    

    predict_y = best_knn.predict(test_X)

    

    #RMSE

    rmse.append(mean_squared_error(test_y,predict_y)**0.5)

    

plt.plot(k_range,rmse)

plt.xlabel('Value of K for KNN')

plt.ylabel('Error')
room2 =  airbnb[(airbnb['price']<175) & (airbnb['room_type']=='Entire home/apt')]

X = room2[['latitude','longitude']].values

y = (room2['price']/10).astype(int)

y = y.values



#split train/test set

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=1/3,random_state=3)



rmse = []

from sklearn.metrics import mean_squared_error

k_range = range(1,21)



for k in k_range:

    #choose best K value

    best_knn = KNeighborsClassifier(n_neighbors=k)

    #train model

    best_knn.fit(train_X,train_y)

    #check score

    print(best_knn.score(test_X,test_y))

    

    predict_y = best_knn.predict(test_X)

    

    #RMSE

    rmse.append(mean_squared_error(test_y,predict_y)**0.5)

    

plt.plot(k_range,rmse)

plt.xlabel('Value of K for KNN')

plt.ylabel('Error')