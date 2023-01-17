import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
imdb = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv",sep=',')

imdb.head()
imdb.rename(columns={'Revenue (Millions)':"Revenue","Runtime (Minutes)":"Runtime"},inplace=True)

print(imdb.columns)
imdb.describe(include='all')
import plotly.express as px

years=imdb.groupby("Year")["Rating"].mean().reset_index()

px.scatter(years,x="Year", y="Rating").show()

mostearned=imdb[imdb["Revenue"]==imdb["Revenue"].max()]

print(mostearned)
imdbtop=imdb[["Title","Director","Rating"]][imdb["Rating"]==imdb["Rating"].max()]

imdbtop.head()
directors=imdb.groupby("Director")["Rating"].mean().reset_index()

directors.sort_values("Rating", ascending=False)



imdb['Rating'].corr(imdb['Metascore'])
import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

matplotlib.style.use('ggplot')



plt.scatter(imdb.Metascore, imdb.Rating)

plt.show()
imdb['Rating'].corr(imdb['Revenue'])
plt.scatter(imdb.Rating, imdb.Revenue)

plt.show()
imdb['Rating'].corr(imdb['Runtime'])
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 6))

ax.set(xscale="log")

sns.scatterplot(imdb.Rating, imdb.Runtime, ax=ax)

plt.show()