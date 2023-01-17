import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/gocoronago/5-4ts.csv")
df
df.info()
df_india = df.loc[df['Country/Region'] == 'India']
df_india = df_india.drop(['Province/State'], axis=1)
df_india["Date"] = pd.to_datetime(df_india['Date'])
df_india
df_india.dtypes
import datetime
df_india = df_india.loc[df_india['Date'] >= '2020-03-01']
df_india.head()
df_india.plot(x="Date", y=["Confirmed", "Recovered", "Deaths"], figsize=(20,5), grid=True)
plt.show()
df_india.plot(x="Date", y=["Confirmed", "Recovered", "Deaths"], figsize=(20,5), grid=True, logy=True)
plt.show()
df = pd.read_csv("/kaggle/input/gocoronago/IndividualDetails.csv")
df
df.info()
df = df.drop(['id'], axis=1)
df["diagnosed_date"] = pd.to_datetime(df['diagnosed_date'])
df
df_state = df['detected_state']
df_state
df_state.value_counts(ascending=True).plot(kind='barh', figsize=(10,10), grid=True)
df_gender = df['gender']
df_gender.value_counts().plot(kind='bar')
from wordcloud import WordCloud, STOPWORDS
df_notes = df['notes'].dropna().str.replace('\d+', '')
text = " ".join(note for note in df_notes)
text
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
df_age = df['age'].dropna().replace('28-35', None).astype(int)
df_age
df_age.hist(bins=20, figsize=(10, 10))
