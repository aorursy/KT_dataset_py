from gensim.models import word2vec

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv").fillna("")

df.head()
count_vectorizer = CountVectorizer(min_df=1) #ngram_range=(1, 2),

analyze = count_vectorizer.build_analyzer()



analyze = count_vectorizer.build_analyzer()

analyze(" Minute Cash cash Payout* £  - Short Te")
#Build text tokenizer

count_vectorizer = CountVectorizer(min_df=1) #ngram_range=(1, 2),

analyze = count_vectorizer.build_analyzer()
df['q1']=df['question1'].apply(lambda x : analyze(x))

df['q2']=df['question2'].apply(lambda x : analyze(x))

#df['common words']=df[['q1','q2']].apply(lambda x, y : list(set(x)&set(y)) )

df[['q1','q2']].head()
corpus = list(df['q1'])+list(df['q2'])
def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=500, workers=4)

tsne_plot(model)