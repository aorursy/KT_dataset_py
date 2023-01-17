#importing necessery libraries for future analysis of the dataset

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns



csv_file = '../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

df = pd.read_csv(csv_file)

df.info()
df.isna().sum()
df[df.last_review.isna()].number_of_reviews.value_counts()
# Drop features `name`, `host_name`, `last_review`

df.drop(columns=['id', 'host_name', 'last_review'], inplace=True)

df.head()
# Fill NaN value with 0 for feature reviews_per_month

df.fillna({'reviews_per_month':0}, inplace=True)

df.head()
df.host_id.value_counts().head(10)
# Prove the count value of 'host_id' equals the value in column 'calculated_host_listings_count'

dfcopy = df.copy()

vcmap = df.host_id.value_counts()

dfcopy['vc'] = dfcopy.apply(lambda row: vcmap[row['host_id']], axis=1)

(dfcopy.vc == dfcopy.calculated_host_listings_count).value_counts()
print("Quantile (95%) of price is:", df.price.quantile(0.95))

df['neighbourhood_group'].unique()
price_400 = df[df.price < 400]

sns.set(rc={'figure.figsize':(11.7,8.27)})

fig = sns.violinplot(data=price_400, x='neighbourhood_group', y='price')

fig.set_title('Density and distribution of prices for each neighberhood_group')
# Find out the price distribution for each neighbourhood group

def qq(x, ratio):

    return x.quantile(ratio)



f = {'price': ['min', 'mean', 'median', 'std', ('25%', lambda v: qq(v, 0.25)), ('50%', lambda v: qq(v, 0.5)), \

               ('75%', lambda v: qq(v, 0.75)), 'max']}

ng = df.groupby('neighbourhood_group')

ng.agg(f)
# Pie usage https://kontext.tech/column/code-snippets/402/pandas-dataframe-plot-pie-chart



f, ax = plt.subplots(1, 2, figsize=(18,8))



# Color representing different neighbourhood.

cs = ['r', 'dodgerblue', 'orange', 'green', 'pink']

df['neighbourhood_group'].value_counts().plot(kind='pie', autopct='%2.2f%%', ax=ax[0], colors=cs, shadow=True)

df['neighbourhood_group'].value_counts().plot(kind='barh', ax=ax[1], color=cs)

ax[0].set_title('Share of Neighbourhood (pie)')

ax[0].set_ylabel('Neighbourhood Share')

ax[1].set_title('Share of Neighbourhood (bar)')

plt.show()
plt.figure(figsize=(10,6))

# To enforce consistence of map color with previous two figures.

color_dict = dict(zip(list(df.neighbourhood_group.value_counts().keys()), cs))

sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', data=df, palette=color_dict)

plt.ioff()
import folium

from folium.plugins import HeatMap

plt.figure(figsize=(12, 6))

m = folium.Map([40.7128, -74.0060], zoom_start=11)

HeatMap(df[['latitude','longitude']].dropna(), radius=8, gradient={.4: 'blue', .65: 'lime', 1: 'red'}).add_to(m)

display(m)



# Or, we can plot a scatter map with 

# fig = price_400.plot(kind='scatter', x='longitude', y='latitude', c='price', 

#                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))
plt.figure(figsize=(10,6))

for i, ng in enumerate(df.neighbourhood_group.unique()):

    sns.distplot(df[df.neighbourhood_group==ng].price, color=cs[i], hist=False, label=ng)



plt.title('Borough wise price destribution for price < 1000')

plt.xlim(0, 1000)

plt.show()
df.room_type.unique()
fig = sns.violinplot(x='room_type', y='price', data=df[df.price < 500])

fig.set_title('Violin plot of room type with respect to price < 500')
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

from PIL import Image

import os, urllib



text = ' '.join(str(n) for n in df.name)

# A New York City map

if not os.path.exists('nyc.jpeg'):

    urllib.request.urlretrieve('http://git.io/JTJqH', 'nyc.jpeg')

    

mask = np.array(Image.open('nyc.jpeg'))

wc = WordCloud(stopwords=STOPWORDS, max_words=200, 

               background_color="white", mask=mask, width=mask.shape[1],

               height=mask.shape[0], max_font_size=64).generate(text)

plt.figure(figsize=(24,12))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
dfcopy = df.copy()

for mn in (3, 10, 30):

    dfcopy['r'] = dfcopy.apply(lambda row: row['minimum_nights'] <= mn, axis=1)

    print(f"minimum_night <= {mn}", dfcopy.r.value_counts(normalize=True), sep="\n")
df.minimum_nights.value_counts().head(30)

sns.distplot(df[df.minimum_nights <= 90]['minimum_nights'], bins=10)
df_review = df[['number_of_reviews', 'reviews_per_month', 'price']]

corr = df_review.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
# Distribution of listings with number_of_reviews <= 20 

# sns.distplot(df.number_of_reviews)

reviews_20 = df[(df.number_of_reviews <= 20) & (df.number_of_reviews > 0)]

fig = reviews_20.plot(kind='scatter', x='longitude', y='latitude', c='number_of_reviews', 

              cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))

# df.number_of_reviews.value_counts().head(20)
for qv in (0.25, 0.5, 0.75):

    print(f'{qv} of listings have no more than', df.number_of_reviews.quantile(qv), 'reviews')
def kde_plot(df):

    plt.figure(figsize=(16,8))

    for i, bor in enumerate(df.neighbourhood_group.unique()):

        sns.kdeplot(df.loc[df['neighbourhood_group'] == bor, 'availability_365'], 

                    color=cs[i], label=bor, shade=True, alpha=0.5)

#         sns.distplot(df.loc[df['neighbourhood_group'] == bor, 'availability_365'])



    plt.xlabel('Availability 365', fontsize=16)

    plt.legend()

    plt.show()

kde_plot(df)