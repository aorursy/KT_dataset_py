# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import folium



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import re

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/auspol2019.csv")

gc= pd.read_csv("../input/location_geocode.csv")

gc.rename(index=str, columns={"name": "user_location"},inplace=True)
df=pd.merge(df, gc, on='user_location')

#fetching the Mentioned accounts

def Mens(m):

    return (re.findall(r"(@.*?)(?:\s)",m))

df["Mentions"]=df.full_text.apply(Mens)
#fetching hashTags

def tags(m):

    return(re.findall(r"(#.*?(?:[\s.]))",m))

df["Tags"]=df.full_text.apply(tags)

    
ndf=df.groupby(["lat","long"])["id"].count()

ndf=ndf.reset_index()
ndf.head()
m = folium.Map(location=[20,0], tiles="Mapbox Bright", zoom_start=2)

for i in range(0,len(ndf)):

    folium.Circle(

      location=[ndf.iloc[i]['long'], ndf.iloc[i]['lat']],

      #popup=ndf.iloc[i]['user_location'],

      radius=ndf.iloc[i]['id']*10000,

      color='crimson',

      fill=True,

      fill_color='crimson'

   ).add_to(m)

m
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
#Finding Scott Morrison
def scott(m):

    if re.search(r"(([@#])?([Ss]cott)?)?\s?[Mm]orrison",m) :

            return True

    else:

        return False

df["Morrison"]=df.full_text.apply(scott)
x=df[df.Morrison==True]

x=x.sort_values("created_at")
sl=[]

for m in x.full_text:

    scores=sid.polarity_scores(m)

    sl.append(scores)
x["Sentiment"]=sl
x.reset_index(inplace=True)
pos=[]

neg=[]

neu=[]

for m in x.Sentiment:

    neg.append(m['neg'])

    pos.append(m['pos'])

    neu.append(m["neu"])

     

        

        
x
day=[]

for m in x.created_at:

    m=m[8:-9]

    day.append(m)

x["day"]=day    
x["neg"]=neg

x["pos"]=pos

x["neu"]=neu
y=x.groupby("day")[["neg","pos","neu"]].mean()
y.reset_index(inplace=True)
plt.figure(figsize=(20,10))

plt.plot( y.day, y.pos, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

plt.plot( y.day, y.neg, marker='o', markerfacecolor='red',markersize=12, color='skyblue', linewidth=4)

#plt.plot( y.day, y.neu, marker='o', markerfacecolor='yellow',markersize=12, color='skyblue', linewidth=4)

plt.legend()
ns=''

for m in x.full_text:

    ns+=''+m
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

ns=re.sub(r'[@#].*?(?:\s)', '', ns)

ns=re.sub(r'http.*?(?:\s)', '', ns)

ns=re.sub(r"(([@#])?([Ss]cott)?)?\s?[Mm]orrison","",ns)

token3=word_tokenize(ns)

#Removing Stopwords

filt3=[]

for r in token3: 

    if len(r)>3:

        if not r.lower() in stop_words: 

            filt3.append(r.lower())
ns=''

for m in filt3:

    ns+=" "+m
from wordcloud import WordCloud

wordcloud = WordCloud(width=1000, height=800, margin=0).generate(ns)

plt.figure(figsize=(20,20))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0,y=0)

plt.show() 