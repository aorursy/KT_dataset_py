import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import wordcloud as wc

%matplotlib inline
df = pd.read_csv(r'../input/amazon_alexa.tsv', delimiter='\t')
df.head()
df.describe()
df.info()
sns.countplot(df["rating"])
plt.figure(figsize=(10,5))

plt.xticks(rotation='vertical')

sns.countplot(df["variation"]),
df_rating_by_variation = df.groupby(["variation","rating"]).count().drop(columns=['verified_reviews', 'feedback'])

df_rating_by_variation.unstack().plot(kind='barh', color = ['r','m','y','c','g'], legend=True, figsize=(18, 18))
mystopwordslist = ["amazon","device","alexa","will","time","sound","one","echo","speaker","now","bought",

              "thing","dot","even","product","devices","set","TV","screen","need","show"]

stopwords = wc.STOPWORDS

for  i in mystopwordslist:

    stopwords.add(i)
reviews = ' '.join(df['verified_reviews'].tolist())

wordcloud = wc.WordCloud(stopwords= stopwords,background_color="black").generate(reviews)

plt.style.use('seaborn-pastel')

plt.figure(figsize=(10, 10))

plt.axis('off')

plt.imshow(wordcloud)

plt.title("Reviews with my stopwords", fontsize = 20)

plt.show()
df1 = df[df["rating"] == 1]

df2 = df[df["rating"] == 2]

df3 = df[df["rating"] == 3]

df4 = df[df["rating"] == 4]

df5 = df[df["rating"] == 5]
reviews = ' '.join(df1['verified_reviews'].tolist())

wordcloud = wc.WordCloud(stopwords= stopwords,background_color="mintcream").generate(reviews)

plt.style.use('seaborn-pastel')

plt.figure(figsize=(8, 8))

plt.axis('off')

plt.imshow(wordcloud)

plt.title("Reviews where rating = 1", fontsize = 20)

plt.show(),

reviews = ' '.join(df2['verified_reviews'].tolist())

wordcloud = wc.WordCloud(stopwords= stopwords,background_color="lightcyan").generate(reviews)

plt.style.use('seaborn-pastel')

plt.figure(figsize=(8, 8))

plt.axis('off')

plt.imshow(wordcloud)

plt.title("Reviews where rating = 2", fontsize = 20)

plt.show(),

reviews = ' '.join(df3['verified_reviews'].tolist())

wordcloud = wc.WordCloud(stopwords= stopwords,background_color="paleturquoise").generate(reviews)

plt.style.use('seaborn-pastel')

plt.figure(figsize=(8, 8))

plt.axis('off')

plt.imshow(wordcloud)

plt.title("Reviews where rating = 3", fontsize = 20)

plt.show(),

reviews = ' '.join(df4['verified_reviews'].tolist())

wordcloud = wc.WordCloud(stopwords= stopwords,background_color="lightblue").generate(reviews)

plt.style.use('seaborn-pastel')

plt.figure(figsize=(8, 8))

plt.axis('off')

plt.imshow(wordcloud)

plt.title("Reviews where rating = 4", fontsize = 20)

plt.show(),

reviews = ' '.join(df5['verified_reviews'].tolist())

wordcloud = wc.WordCloud(stopwords= stopwords,background_color="lightsteelblue").generate(reviews)

plt.style.use('seaborn-pastel')

plt.figure(figsize=(8, 8))

plt.axis('off')

plt.imshow(wordcloud)

plt.title("Reviews where rating = 5", fontsize = 20)

plt.show()