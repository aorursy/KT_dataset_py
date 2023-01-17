from os import path

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt



from wordcloud import WordCloud, STOPWORDS



from nltk.corpus.reader import XMLCorpusReader

reader = XMLCorpusReader('../input/', 'BDS00389.xml')

print(reader)

print(reader.raw('BDS00389.xml'))

print(reader.fileids())

print(reader.words())

str1 = ' '.join(reader.words())

text = ((str1.encode("utf-8")).decode("utf-8"))

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