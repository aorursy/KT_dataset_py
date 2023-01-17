import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE
df = pd.read_csv('../input/alcohol-consumption-in-russia/russia_alcohol.csv')
df.head()
df.describe()
g=df.hist(figsize=(20,15))
mask=np.zeros_like(df.corr(), dtype=bool)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(df.corr(), mask=mask)
df_transformed=pd.DataFrame(columns=['year','region','alco','sale'])

for index,row in df.iterrows():

    s = pd.Series([row['year'], row['region'], 'wine', row['wine']],df_transformed.columns)

    s1 = pd.Series([row['year'], row['region'], 'beer', row['beer']],df_transformed.columns)

    s2 = pd.Series([row['year'], row['region'], 'vodka', row['vodka']],df_transformed.columns)

    s3 = pd.Series([row['year'], row['region'], 'champagne', row['champagne']],df_transformed.columns)

    s4 = pd.Series([row['year'], row['region'], 'brandy', row['brandy']],df_transformed.columns)

    df_transformed= df_transformed.append([s,s1,s2,s3,s4],ignore_index=True) 
df_transformed
g=sns.relplot(x="year", y="sale",

            hue="alco",

            kind="line", data=df_transformed,size=10, aspect=2);
top_beer_region=df_transformed[df_transformed['alco']=='beer'].groupby('region').mean()

top_beer_region=top_beer_region.reset_index().sort_values(by='sale',ascending=False)

plt.figure(figsize=(15,7))

g=sns.barplot(data=top_beer_region[:10],y='sale',x='region')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.tick_params(labelsize=10)
top_wine_region=df_transformed[df_transformed['alco']=='wine'].groupby('region').mean()

top_wine_region=top_wine_region.reset_index().sort_values(by='sale',ascending=False)

plt.figure(figsize=(15,7))

g=sns.barplot(data=top_wine_region[:10],y='sale',x='region')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.tick_params(labelsize=10)
top_vodka_region=df_transformed[df_transformed['alco']=='vodka'].groupby('region').mean()

top_vodka_region=top_vodka_region.reset_index().sort_values(by='sale',ascending=False)

plt.figure(figsize=(15,7))

g=sns.barplot(data=top_vodka_region[:10],y='sale',x='region')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

g.tick_params(labelsize=10)