import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# The dataset has 2 different types of data files available

# Let's see what is the difference between them before we proceed..

wine_130k = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)

wine_150k = pd.read_csv("../input/winemag-data_first150k.csv", index_col=0)
wine_130k.info()
wine_150k.info()
# So the data file with 130k reviews has 3 more columns than the 150k file, let's see what extra columns are there..

for column in wine_130k.columns:

    if(column not in wine_150k.columns):

        print(column)
wine_130k.head()
wine_130k.describe()

# We have only two numerical values in the dataset: points, price
wine_130k.rename(index=str, columns={"designation":"vineyard", "variety":"grape_variety"}, inplace=True)

wine_130k.info()
wine_130k.loc[wine_130k.country.isnull()]
wine_130k.groupby('country').winery.value_counts()
country_by_winery = wine_130k.groupby('winery').country.unique().to_json()

import json

country_by_winery = json.loads(country_by_winery)
country_by_winery
# Function for getting country respective to the winery

def GetCountry(w):

    cntry = country_by_winery[w]

    if(None in cntry):

        cntry = cntry.remove(None)

    return cntry[0] if cntry else np.NaN
#GetCountry('Famiglia Meschini')

country_by_winery['Ross-idi']
wine_130k.country.value_counts()
wine_130k['country_by_winery'] = wine_130k.winery.map(GetCountry)

# US has the most number of wineries

wine_130k.country_by_winery.fillna('US', inplace=True)

wine_130k.country.fillna(wine_130k.country_by_winery, inplace=True)

wine_130k.drop('country_by_winery', axis=1, inplace=True)
wine_130k.price.value_counts()
log_price = np.log(wine_130k.price)

log_price.plot.hist(title='Price distribution histogram', bins=20, color='tomato', edgecolor='black', figsize=(10,7));
sns.set(style="whitegrid")

plt.figure(figsize=(7,7))

sns.boxplot(wine_130k.price);
wine_130k.groupby('country').price.median().plot.line(title='Median distribution of price by country', fontsize=60, figsize=(100,30), linewidth=3.3, color='red');
median_price = wine_130k.groupby('country').price.transform('median')

wine_130k.price.fillna(median_price, inplace=True)
wine_130k[wine_130k.price.isnull()]
# 'Egypt' has only one review and has no price value, using overall price median to fill

wine_130k.price.fillna(wine_130k.price.median(), inplace=True)
wine_130k[wine_130k.taster_name.isnull()]
wine_130k['taster_name'].value_counts()
# List of countries with taster that has maximum reviews in that specific country

taster_by_country = wine_130k.groupby('country').apply(lambda x: x['taster_name'].value_counts().index[0])

taster_by_country
wine_130k['taster_by_country'] = wine_130k[wine_130k.taster_name.isnull()].country.map(taster_by_country)

wine_130k.taster_name.fillna(wine_130k['taster_by_country'], inplace=True)

wine_130k.drop('taster_by_country', axis=1, inplace=True)
wine_130k[wine_130k.grape_variety.isnull()]
wine_130k[wine_130k.winery=='Carmen'].grape_variety.value_counts()
# Filling with the max number of variety

wine_130k.grape_variety.fillna('Cabernet Sauvignon', inplace=True)
# How the dataset looks like now:

wine_130k.info()
wine_130k.describe()
print('Avegare Wine price: ', wine_130k.price.mean())

print('Median Wine price: ', wine_130k.price.median())
wine_130k.taster_name.value_counts()
reviews_per_taster = wine_130k['taster_name'].value_counts().plot.bar(title='Review Distribution by Taster', fontsize=14, figsize=(12,8), color='tomato');

plt.xlabel('Taster')

plt.ylabel('Number of reviews')

plt.show()
reviews_per_country = wine_130k['country'].value_counts().plot.bar(title='Review Distribution by Countries', fontsize=14, figsize=(17,10), color='c');

plt.xlabel('Country')

plt.ylabel('Number of reviews')

plt.show()
plt.figure(figsize=(25,10))

wine_130k.groupby('country').max().sort_values(by="points",ascending=False)["points"].plot.bar(fontsize=17)

plt.xticks(rotation=70)

plt.xlabel("Country of Origin")

plt.ylabel("Highest point of Wines")

plt.show()
price_box = wine_130k.price.plot.box(title='Price Boxplot', figsize=(10,7));
price_hist = wine_130k.price.plot.hist(title='Price Distribution by intervals', bins=30 ,figsize=(7,7));

plt.xlabel('Price')

plt.ylabel('Reviews')

plt.show()
print("Skewness of Price: %f" % wine_130k['price'].skew())
log_price_hist = log_price.plot.hist(title='Price(log) distribution histogram', bins=20, color='orange', edgecolor='black', figsize=(10,7));

plt.xlabel('Price')

plt.ylabel('Reviews')

plt.show()
points_bar = wine_130k['points'].value_counts().sort_index().plot.bar(title='Points Distribution', figsize=(15,10), color='firebrick')

plt.xlabel('Points')

plt.ylabel('Number of reviews')

plt.show()
print('Min points:', wine_130k.points.min())

print('Max points:', wine_130k.points.max())
sns.violinplot(wine_130k['price'], wine_130k['taster_name'], figsize=(15,15)) #Variable Plot

sns.despine()
fig = plt.figure(figsize=(14,8))

ax = fig.add_subplot(1,1,1)

ax.scatter(wine_130k['points'],wine_130k['price'], alpha=0.15)

plt.title('Points vs Price')

plt.xlabel('Points')

plt.ylabel('Price')

plt.show()
plt.figure(figsize=(15,10))

price_points_box = sns.boxplot(x="points", y="price", data = wine_130k);

plt.title('Points Boxplot')

plt.show()
plt.rc('xtick', labelsize=20)     

plt.rc('ytick', labelsize=20)

fig = plt.figure(figsize=(30,24))

ax = fig.add_subplot(1,1,1)

province_not_null = wine_130k[wine_130k.province.notnull()]

ax.scatter(province_not_null['points'],province_not_null['country'], s=province_not_null['price'])

plt.xlabel('Points')

plt.ylabel('Country')

plt.show()
df1= wine_130k[wine_130k.grape_variety.isin(wine_130k.grape_variety.value_counts().head(5).index)]

plt.figure(figsize=(15,10))

sns.boxplot(

    x = 'grape_variety',

    y = 'points',

    data = df1,     

);
sns.set()

columns = ['price', 'points']

plt.figure(figsize=(20,20))

sns.pairplot(wine_130k[columns],size = 10 ,kind ='scatter',diag_kind='kde')

plt.show()
wine_130k.pivot_table(index='country', columns='taster_name', values='points', aggfunc='mean')
from wordcloud import WordCloud, STOPWORDS

from PIL import Image

import requests

import urllib
wine_mask = np.array(Image.open(requests.get("http://www.clker.com/cliparts/4/7/5/6/11949867422027818929wine_glass_christian_h._.svg.hi.png", stream=True).raw))
title_text = " ".join(t for t in wine_130k['title'])
title_wordcloud = WordCloud(width = 1024, height = 1024, background_color='#D1310F', mask=wine_mask, contour_width=1, contour_color='black').generate(title_text)
plt.figure( figsize=(30,15), facecolor = 'white' )

plt.imshow(title_wordcloud, interpolation='bilinear')

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
# title_wordcloud.to_file("title.png")
review_text = " ".join(review for review in wine_130k.description)

stopwords = set(STOPWORDS)

stopwords.update(["drink", "now", "wine", "flavor", "flavors"])
reviews_wordcloud = WordCloud(width=1024, height=1024, stopwords=stopwords, max_words=1000, background_color="#D8F2EE").generate(review_text)
plt.figure( figsize=(30,15), facecolor = 'white' )

plt.imshow(reviews_wordcloud, interpolation='bilinear')

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
#reviews_wordcloud.to_file("reviews.png")
mean_points = wine_130k.groupby('country').points.mean()

mean_points
mean_points.plot.line(title='Mean points per Country', figsize=(15,7));
top_vineyards = wine_130k.groupby('vineyard').points.median().nlargest(15)
top_vineyards.plot.line(rot=80, title='Top rated Wines', figsize=(10,7), fontsize=12, color='tomato');
costly_vineyards = wine_130k.groupby('vineyard').price.median().nlargest(15)

costliest_wine = costly_vineyards.plot.bar(figsize=(10,7), fontsize=12, color='firebrick')

plt.title('Costliest Wines by Vineyard')

plt.show()
costly_province = wine_130k.groupby('province').price.median().nlargest(15)

costliest_wine_by_province = costly_province.plot.bar(figsize=(10,7), fontsize=12, color='c')

plt.title('Costliest Wines by Province')

plt.show()
costly_region = wine_130k.groupby(['region_1', 'region_2']).price.median().nlargest(15)

costliest_wine_by_region = costly_region.plot.bar(figsize=(10,7), fontsize=12, color='skyblue')

plt.title('Costliest Wines by Region')

plt.show()
costly_title = wine_130k.groupby(['title']).price.median().nlargest(15)

costliest_wine_by_title = costly_title.plot.area(rot=80, figsize=(10,7), fontsize=12, color='pink')

plt.title('Costliest Wines by Title')

plt.show()
# f, (ax1,ax2) = plt.subplots(1, 2, figsize=(50,20))



top_grape_points = wine_130k.groupby('grape_variety').points.median().nlargest(15)

pl1 = top_grape_points.plot.bar(figsize=(10,7), fontsize=12, color='firebrick')

plt.title('Top Rated Wine by Grape Variety')

plt.ylabel('Median Points')

plt.show()
top_grape_price = wine_130k.groupby('grape_variety').price.median().nlargest(15)

pl2 = top_grape_price.plot.bar(figsize=(10,7), fontsize=12, color='orange')

plt.title('Top Rated Wine by Grape Variety')

plt.ylabel('Median Price')

plt.show()
top_winery_vineyard = wine_130k.groupby(['winery', 'vineyard']).price.median().nlargest(15)

costliest_wine_winery_vineyard = top_winery_vineyard.plot.line(rot=90, figsize=(10,7))

plt.title('Top Priced Wine by Winery and Vineyard')

plt.ylabel('Median Price')

plt.show()
top_winery_vineyard