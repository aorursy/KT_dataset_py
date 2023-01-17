import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
df=pd.read_csv('../input/indian-startup-funding/startup_funding.csv',index_col=0)

df.head()
df['City  Location'] = df['City  Location'].replace(['Bangalore','Bengaluru'],'Bangalore')

df['City  Location'] = df['City  Location'].replace(['Gurgaon','Gurugram'],'Gurugram')



df['InvestmentnType'] = df['InvestmentnType'].replace(['Seed/ Angel Funding',

                                                       'Seed / Angel Funding',

                                                       'Seed/Angel Funding',

                                                       'Angel / Seed Funding',

                                                       'Seed Funding'],'Seed / Angel Funding')

df["Amount in USD"] = df["Amount in USD"].apply(lambda x: str(x).replace(",",""))

df["Amount in USD"] = pd.to_numeric(df["Amount in USD"],errors='coerce')
print('Shape of data',df.shape)

df.describe()
Investment=df.InvestmentnType.value_counts()

plt.figure(figsize=(15,12))

plt.subplot(221)

g = sns.barplot(x=Investment.index[:10],y=Investment.values[:10])

g.set_xticklabels(g.get_xticklabels(),rotation=70)

g.set_xlabel("Investment Types", fontsize=15)

g.set_ylabel("No of fundings made", fontsize=15)

plt.show()
city=df['City  Location'].value_counts()

plt.figure(figsize=(15,12))

plt.subplot(221)

g = sns.barplot(x=city.index[:10],y=city.values[:10])

g.set_xticklabels(g.get_xticklabels(),rotation=45)

g.set_xlabel("Cities", fontsize=15)

g.set_ylabel("No of fundings made", fontsize=15)

plt.show()
maxfundingsdf=df.sort_values(by='Amount in USD',ascending=False,na_position='last')

top_fundings=maxfundingsdf['Amount in USD'].head(5)

invester_names= maxfundingsdf['Investors Name'].head(5)

plt.figure(figsize=(15,12))

plt.subplot(221)

g = sns.barplot(x=invester_names,y=top_fundings)

g.set_xticklabels(g.get_xticklabels(),rotation='vertical')

g.set_xlabel("\nInvesters group", fontsize=15)

g.set_ylabel("Max Amount", fontsize=15)

valuecount_investers=df['Investors Name'].value_counts()

plt.subplot(222)

g1 = sns.barplot(x=valuecount_investers.index[:5], y=valuecount_investers.values[:5]) 

g1.set_xticklabels(g.get_xticklabels(),rotation='vertical')

g1.set_xlabel("\nInvesters group", fontsize=15)

g1.set_ylabel("number of fundings made", fontsize=15)



plt.show()
plt.figure(figsize = (15,15))



stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=150,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(df[df['Industry Vertical'] == 'Technology']['Investors Name']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
plt.figure(figsize = (15,15))



stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=150,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(df['Startup Name']))



print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()