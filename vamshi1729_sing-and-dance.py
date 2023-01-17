# This is my first kaggle kernel, any suggestions and modifications are welcome. Thank you!
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("../input/songdata.csv")
df.describe()
df["text"][0]
# I guessed MJ lyrics could have been included. My guess turned out to be true
MJ_dataframe = df[df["artist"].str.contains("Michael Jackson")]
print (MJ_dataframe)
print (MJ_dataframe.shape)
stop = set(stopwords.words('english'))
final_set = []
for row in MJ_dataframe.iterrows():
    final_set += list(set(TextBlob(row[1][3]).words)-stop)
text = ""
for word in final_set:
    text += word+" "
wordcloud = WordCloud().generate(text)

wordcloud = WordCloud(background_color='white',max_words=200,max_font_size=40, scale=3,).generate(str(text))
fig = plt.figure(1, figsize=(18, 18))
plt.axis('off')

plt.imshow(wordcloud)
plt.show()
polarity_list = []
for i in range(df.shape[0]):
    blob1 = df["text"][i]
    polarity_list.append(TextBlob(blob1).sentiment.polarity)
polarity_list[0]
se = pd.Series(polarity_list)
df["sentiment_polarity"]=se.values
language_df = df[df["sentiment_polarity"]==0]
language_df
from langdetect import detect
# making a language list assuming that sentiments for "non english" are not handled by TextBlob.

language_list = []
for i in range(df.shape[0]):
    if(df["sentiment_polarity"][i]==0):
        iso_string = detect(df["text"][i])
        language_list.append(iso_string)
    else:
        language_list.append("en")
# added languages column to the dataframe
df["languages"]=pd.Series(language_list).values
# lyrics in english, Tagalog,Hindi,French,... used package iso639 from languages (this converts "en" to English locale)
df["languages"].value_counts()
## i dont know why lata mangeshkar songs are detected by isocode "so".
# in my local computer i used TextBlob detect_language() method to  find the language and it gave me correct results
# detect_language() method is showing errors on kaggle.(except OSError as err: # timeout error URL error ) 
# hence i switched to langdetect package from TextBlob
df[df["languages"]=="so"]
