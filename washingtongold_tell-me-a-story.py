import pandas as pd

data = pd.read_csv('/kaggle/input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv')

data.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(25,5))

sns.distplot(data['Plot'].apply(len))
string = 'Hi.I am a chicken, num [1]'

blacklist = []

for i in range(100):

    blacklist.append('['+str(i)+']')

def remove_brackets(string):

    for item in blacklist:

        string = string.replace(item,'')

    return string
data['Plot'] = data['Plot'].apply(remove_brackets)
import gensim

from textblob import TextBlob

def summary(x):

    if len(x) < 500 or str(TextBlob(x).sentences[0]) == x: #under 500 characters or only one sentence

        return x

    else:

        try:

            return gensim.summarization.summarize(x)

        except:

            print("ERROR")

            print("TEXT LENGTH:",len(x))

            print(x)

            return x

data['Summary'] = data['Plot'].apply(summary)
data.to_csv('data.csv',index=False)