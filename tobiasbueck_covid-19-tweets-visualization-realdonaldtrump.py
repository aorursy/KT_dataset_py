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
df = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv")
df.head()
text = list(df["text"])
from nltk.tokenize import TweetTokenizer
words = []
tweet_tok = TweetTokenizer()
print(text[0])
print(tweet_tok.tokenize(text[0]))
for t in text:
    words.extend(tweet_tok.tokenize(t))
print(words[:10])
unique_words = list(set(words))
count_word = {uw:0 for uw in unique_words}
for w in words:
    count_word[w]+=1
list(count_word.items())[:10]
wc = list(count_word.items())
wc_sorted = list(reversed(sorted(wc, key=lambda x: x[1])))
count_word_sorted = {k:v for k,v in wc_sorted}

wc_sorted[:10]
import nltk
tagged = nltk.pos_tag(list(map(lambda x: x[0], wc_sorted)))
allowed_tags = ["FW", "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS"]
allowed_words =list(filter(lambda x: x[1] in allowed_tags ,tagged))
allowed_words[:10]
allowed_words2 = list(filter(lambda x: len(x[0]) > 1, allowed_words))
print(allowed_words2[:10])
aw_count = []
for aw in allowed_words2:
    aw_count.append((aw[0], count_word_sorted[aw[0]]))
aw_sorted = list(reversed(sorted(aw_count, key=lambda x: x[1])))
aw_sorted[:30]
from wordcloud import WordCloud
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
mask = np.array(Image.open("/kaggle/input/covidimage/download.png"))
mask[380:400]
from random import randint
def get_color(word, font_size, position, orientation, font_path, random_state):
    return 150 + randint(0,100), 0 + randint(0,100), randint(0, 50)
wordcloud = WordCloud(mask=mask, width=800, height=600, background_color="white", color_func=get_color)
# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
text = {k:v for k,v in aw_sorted}

wordcloud.generate_from_frequencies(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
fig, ax = plt.subplots(figsize=(60, 35))
ax.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
fig.savefig("covid_wordcloud3.png")