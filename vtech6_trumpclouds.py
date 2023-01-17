# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bs4 import BeautifulSoup

import requests

import re

from wordcloud import WordCloud

from PIL import Image

import random

import matplotlib.pyplot as plt

from matplotlib import cm

from nltk import regexp_tokenize

from nltk.corpus import stopwords

import nltk

from io import BytesIO

import urllib.request

from wordcloud import ImageColorGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tweets = pd.read_csv(r'/kaggle/input/trump-tweets/trumptweets.csv')
tweets.head()
content = tweets['content']
content[:10]
sw = stopwords.words('english')

patn = '\w+'

blacklist = [

    '[document]',

    'noscript',

    'header',

    'html',

    'meta',

    'via'

    'head',

    'input',

    'solo',

    'band',

    'guitars',

    'vocals',

    'guitar',

    'bass',

    'song',

    'writer',

    'composed',

    'composer',

    'music',

    'submit',

    'site',

    'request',

    'ask',

    'send',

    'like',

    'share',

    'correcting',

    'correct',

    'correction',

    'thank',

    'thanks',

    'you',

    'more',

    'http',

    'realdonaldtrump',

    'https',

    'com',

    'www',

    'twitter',

]



def data_cleanup(text):

    text = text.lower()

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'([.!,?])', r' \1 ', text)

    text = re.sub(r'[^a-zA-Z.,!?]+', r' ', text)

    text = regexp_tokenize(text, patn)

    text = [i for i in text if (i not in sw) and (i not in blacklist)]

    return text



clean_text = content.apply(lambda x: data_cleanup(x))
words=[]

for i in clean_text:

    for j in i:

        words.append(j)

        

word_freq = nltk.FreqDist([i for i in words if len(i) > 2])
plt.figure(figsize=(16, 6))

word_freq.plot(50)
donald = 'https://www.cbc.ca/interactives/content/lead_images/_lead-large/hero-altamerica.jpg'

with urllib.request.urlopen(donald) as url:

    f = BytesIO(url.read())

img = Image.open(f)



mask = np.array(img)

img_color = ImageColorGenerator(mask)



wc = WordCloud(background_color='white',

              mask=mask,

              max_font_size=2000,

              max_words=2000,

              random_state=42)

wcloud = wc.generate_from_frequencies(word_freq)

plt.figure(figsize=(16, 10))

# recolor wordcloud and show

# we could also give color_func=image_colors directly in the constructor

plt.axis('off')

plt.imshow(wc.recolor(color_func=img_color), interpolation="bilinear")

plt.show()
donald2 = 'https://ih0.redbubble.net/image.208240824.9874/flat,750x,075,f-pad,750x1000,f8f8f8.jpg'

with urllib.request.urlopen(donald2) as url:

    f = BytesIO(url.read())

img = Image.open(f)



mask = np.array(img)

img_color = ImageColorGenerator(mask)



wc = WordCloud(background_color='white',

              mask=mask,

              max_font_size=2000,

              max_words=2000,

              random_state=42)

wcloud = wc.generate_from_frequencies(word_freq)

plt.figure(figsize=(10, 16))

plt.axis('off')

# recolor wordcloud and show

# we could also give color_func=image_colors directly in the constructor

plt.imshow(wc.recolor(color_func=img_color), interpolation="bilinear")

plt.show()