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
df = pd.read_csv('/kaggle/input/russian-touris-attractions/tourist_attractions.csv')
df.head(10)
df['region'].value_counts().head(20).plot.bar() #top 20 russian regions by count of tourist attracktions
from wordcloud import WordCloud

import matplotlib.pyplot as plt



def wordcoluder(col):

    words = []

    new = col.split(" ")

    for i in range(len(new)):

        words.append(new[i])

    return words



df['words'] = df['name'].apply(wordcoluder)



text = " ".join(review for review in df.name)

print ("There are {} words in the combination of all review.".format(len(text)))



wordcloud = WordCloud( background_color="white").generate(text)



plt.figure(figsize=[15,20])

plt.title("Words of Russian Tourist attracktions",size= 30)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()