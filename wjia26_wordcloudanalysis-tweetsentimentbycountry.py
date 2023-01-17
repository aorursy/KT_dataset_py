# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
df=pd.read_csv('../input/twittersentimentbycountry/TwitterSentimentByCountry.csv')
def word_freq_generator(dfWordCloud):
    ##Count frequencies of words
    ##Removes the stop words and characters
    stop_words = set(stopwords.words('english'))
    stop_char=['#','@','&']
    stop_words.update(["rt","https","https.","-",'.',':'])
    dfWordCloud
    text = " ".join(tweet.lower() for tweet in dfWordCloud.text)
    all_freq={}
    for word in text.split():
        res = [char for char in stop_char if(char in word)]
        if len(res)==0:
            if word not in stop_words:
                if word in all_freq: 
                    all_freq[word] += 1
                else: 
                    all_freq[word] = 1  
    return all_freq
## Change this for a different country
dfCountry=df[df['file_name']=='HongKong']
# dfCountry=df[df['file_name']=='Australia']
all_freq=word_freq_generator(dfCountry)
s = pd.Series(all_freq, name='count')
s.index.name = 'word'
sdf=s.reset_index()
sdf=sdf.sort_values('count',ascending=False)
fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
sdf[0:20].sort_values(by='count').plot.barh(x='word',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Tweets (Without Stop Words)")

plt.show()
##Generate wordcloud from frequencies
wordcloud = WordCloud(stopwords=stopwords,background_color="white").generate_from_frequencies(all_freq)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
dfWordCloud['polarity'].mean()
