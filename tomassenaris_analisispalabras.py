# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install youtube_transcript_api
!pip install pystemmer
!python -m spacy download es
!pip install pillow
!pip install wordcloud
!pip install nltk
!python -m nltk.downloader all
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize, wordpunct_tokenize
import Stemmer
import spacy
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from youtube_transcript_api import YouTubeTranscriptApi
def getVideoTranscript(id, langs=['es']):
    transcript = YouTubeTranscriptApi.get_transcript(id, languages=langs)
    text =  ' '.join(map(lambda x: x['text'], transcript))
    return [word for word in wordpunct_tokenize(text)]
def dropStopWords(words):
    stopw = stopwords.words('spanish')
    stopw.extend(['.', '[', ']', ',', ';', '', ')', '),', ' ', '(', '?', '¿', ':', '"', '".', '--'])
    return [token for token in words if token not in stopw]


def summarize(words):
    # remove stop words
    words = dropStopWords(words)

    dict = {}
    for w in words:
        dict[w] = 1 + (dict[w] if w in dict else 0)

    result = pd.DataFrame.from_dict(dict, orient='index').reset_index().rename(columns={0: 'count', 'index': 'word'})
    
    stemmer = Stemmer.Stemmer('spanish')
    result['stem'] = stemmer.stemWords(result['word'])

    return result
def histogram(df, x_col, y_col):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_axes([0,0,1,1])
    ax.bar(df[x_col], df[y_col])
    plt.xticks(rotation=90, fontsize=14)
    plt.show()
#videoId = 'cOKyYlxLWPw&t' # UkiDean
#videoId = 'h9VNOWg5KZA' # PicoPala
videoId = 'cVwAw85OsBc'
words = getVideoTranscript(videoId)
df = summarize(words).sort_values(by='count', axis=0, ascending=False)
histogram(df[0:50], 'word', 'count')
wc = WordCloud(max_font_size=100, max_words=50, background_color="white").generate(' '.join(dropStopWords(words)))
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()