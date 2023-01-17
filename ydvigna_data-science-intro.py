import numpy as np

import pandas as pd

import os



pd.set_option('display.max_colwidth', -1)



# Verificando se arquivo 'winemag-data-130k-v2.csv' foi encontrado

print(os.listdir("../input"))

fruit_sales = pd.DataFrame({'Apples': [35, 41, 44, 45, 50, 100, 15], 

                            'Bananas': [21, 34, 20, 14, 19, 80, 4]}, 

                            index=['2013 Sales', '2014 Sales', '2015 Sales', '2016 Sales', 

                                   '2017 Sales', '2018 Sales', '2019 Sales'])

fruit_sales
fruit_sales.describe()
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)

reviews.head()
len(reviews.index)
reviews.describe()
reviews.groupby(['country']).points.mean().sort_values(ascending=False).to_frame()
reviews.groupby(['country']).price.mean().sort_values(ascending=False).to_frame()
reviews.loc[reviews['points'].idxmax()]
reviews.loc[reviews['price'].idxmax()]
points_to_price_ratio = reviews.points / reviews.price

id_best_ratio = points_to_price_ratio.idxmax()

bargain_wine = reviews.loc[id_best_ratio, 'title']

print(bargain_wine)
reviews.loc[reviews.title == 'Bandit NV Merlot (California)']
reviews.description[1]
reviews.iloc[0]
df = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'variety']]

df
print(*reviews.country.dropna().unique(), sep='\n')

reviews.country.value_counts().to_frame()
reviews.loc[reviews.country == 'Italy']
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])

countries_reviewed
country_variety_counts = reviews.groupby(['country', 'variety']).description.count().sort_values(ascending=False)

country_variety_counts.to_frame()
tropical_count = reviews.description.map(lambda desc: "tropical" in desc).sum()

fruity_count = reviews.description.map(lambda desc: "fruity" in desc).sum()



descriptor_counts = pd.Series([tropical_count, fruity_count], index=['tropical', 'fruity'])

descriptor_counts.to_frame('count')
reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

reviews_written.sort_values(ascending=False).to_frame()
reviewer_mean_ratings = reviews.groupby('taster_twitter_handle').points.mean()

reviewer_mean_ratings.sort_values(ascending=False).to_frame()