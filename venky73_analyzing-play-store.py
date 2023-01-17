import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
df = pd.read_csv("../input/googleplaystore_user_reviews.csv")

#print(os.listdir("../input/googleplaystore_user_reviews.csv"))
df.sample()
df.isna().sum()
df.sample(5)
df = df[~df.Sentiment.isna()]

df.isna().sum()
len(df)
df.Sentiment.value_counts()
scores = df.Sentiment.map({"Positive":1,"Negative":-2,"Neutral":0})

df.Sentiment = scores
df.sample(3)
df.drop(columns=['Translated_Review','Sentiment_Polarity','Sentiment_Subjectivity'],inplace=True)
df.head()
after_editing = df.groupby('App').sum().reset_index()

after_editing.sample(5)
after_editing.sort_values(by='Sentiment',ascending=False)[:10]
after_editing.sort_values(by='Sentiment')[:10]
df = pd.read_csv("../input/googleplaystore.csv")
df.sample()
df.shape
len(df.App.unique())
df[df.App == "Angry Birds Classic"]
df.drop_duplicates(inplace=True)
len(df)
df.drop(columns=['Current Ver','Android Ver','Content Rating','Genres','Price'], inplace = True)
df.sample(5)
df.dropna(inplace= True)
df.isna().sum()
len(df.App.unique())
df.shape
len(after_editing)
df['Sentiment'] = np.NaN
x = list(after_editing.App)

for i in range(len(df)):

    check = df.iloc[i,0]

    if check in x:

        df.iloc[i,-1] = int(after_editing[after_editing.App == check].Sentiment)
df[df.App == '10 Best Foods for You']
after_editing[after_editing.App == '10 Best Foods for You']
h = int(after_editing[after_editing.App == '10 Best Foods for You'].Sentiment)

h
df.isna().sum()
df.head(5)
df[df.App=="ROBLOX"]
df.drop(columns='Category',inplace=True)
len(df)
df.drop_duplicates(inplace=True)
len(df)
df.drop_duplicates(subset=['App'],inplace=True)
df[df.App=="8 Ball Pool"]
len(df)
df.isna().sum()
df.info()
df.sort_values(by='Rating')
df.sort_values(by='Sentiment',ascending=False)[:5]
df[~df.Sentiment.isna()].sort_values(by='Sentiment')[:5]
df.sort_values(by='Rating')[:5]
df.sort_values(by='Rating')[:-6:-1]

#As we see rating is more than 5.0 which is an outlier. So delete that record
df = df[df.Rating != 19.0]
df.sort_values(by='Rating')[:-6:-1]
df['Reviews'].sample(10)
df = df[df.Reviews.astype('int') > 1000 ]
df.sample(5)