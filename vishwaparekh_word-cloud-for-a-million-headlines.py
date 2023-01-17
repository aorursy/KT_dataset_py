# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from nltk.corpus import stopwords;



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Read the data

data = pd.read_csv("../input/abcnews-date-text.csv",error_bad_lines=False,usecols =["headline_text"])

# Visualize the first few rows

data.head()
#Create a word cloud

#stopwords1 = set(STOPWORDS)

stopwords1 = stopwords.words()

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords1,

                          max_words=500,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['headline_text']))

fig = plt.figure(figsize=(20,20))

#fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
