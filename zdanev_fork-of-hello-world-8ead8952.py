import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_json("../input/train.json")

df.head()
df['num_photos'] = df['photos'].map(len)

df['num_features'] = df['features'].apply(len)

df['num_description_words'] = df['description'].apply(lambda x: len(x.split(' ')))
features = pd.DataFrame({'feature': [j for i in df.features.values for j in i]})

features['dummy'] = 1

top_features = features.groupby('feature').count().sort_values('dummy', ascending=False).head(100)
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "num_photos", "num_features", "num_description_words"]



for x in top_features.index:

    fn = x.lower().replace(' ', '_').replace('/', '_').replace('-', '_');

    df[fn] = df['features'].map(lambda a: x in a);

    features_to_use.append(fn)

    

print(features_to_use)
df.groupby('interest_level').listing_id.count().loc[['low', 'medium', 'high']].plot(kind='bar')

plt.show()
df.groupby('bedrooms').listing_id.count().plot(kind='bar')

plt.show()
df.groupby('bathrooms').listing_id.count().plot(kind='bar')

plt.show()
df.price.describe(percentiles=[0.01, 0.99])
plt.figure(figsize=(8,6))

sns.distplot(df.price.values, bins=50, kde=True)

plt.xlabel('price')

plt.show()