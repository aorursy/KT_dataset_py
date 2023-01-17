from os import path

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter

from nltk.corpus.reader import XMLCorpusReader

from wordcloud import WordCloud, STOPWORDS
reader = XMLCorpusReader('../input/', 'BDS00389.xml')
print(reader.fileids())
print(reader)
print(reader.raw('BDS00389.xml'))
print(reader.words())
str1 = ' '.join(reader.words())

print(str1)
text = (str1.encode("utf-8").decode("utf-8"))

print(text)
stopwords = set(STOPWORDS)



wc = WordCloud(background_color="white", max_words=2000, 

               stopwords=stopwords)

# generate word cloud

wc.generate(text)



# show

plt.imshow(wc)

plt.axis("off")

plt.figure()

plt.axis("off")

plt.show()
counter = Counter(reader.words())

print(counter.most_common(50))