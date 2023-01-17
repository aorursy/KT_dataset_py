# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image

import urllib.request

from skimage import io

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/twitter-seo/twitter_seo.csv")

df.head()
print("There are {} observations and {} features in this dataset. \n".format(df.shape[0],df.shape[1]))



print("There are {} authors in this dataset such as {}... \n".format(len(df.author.unique()),

                                                                           ", ".join(df.author.unique()[0:5])))
url = df['profile'].iloc[100]



io.imshow(io.imread(url))

io.show()
author = df[['id','author']].groupby('author').count().sort_values(by = 'id', ascending = False)

author.head(10)
# Start with one review:

text = df.text[20]



# Create and generate a word cloud image:

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()