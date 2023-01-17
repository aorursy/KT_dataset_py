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
train_data = pd.read_csv("../input/train.csv")
questions_set = set(train_data['question1']).union(train_data['question2'])
stoplist = set('for a of the and to in is an'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in questions]
from collections import defaultdict

frequency = defaultdict(int)

for text in texts:

    for token in text:

        frequency[token] += 1        
texts = [[token for token in text if frequency[token] > 1] for text in texts]
from pprint import pprint  # pretty-printer
from gensim.models import Word2Vec
min_count = 2

size = 50

window = 4

 

model = Word2Vec(texts, min_count=min_count, size=size, window=window)
model.wv['how']
model.wv.most_similar('flipkart')
model.wv.similarity('men', 'tree')
word_vectors = model.wv
def get_vector_for_sentence(sentence):

    vec = np.zeros(50)

    words = [word for word in sentence.lower().split() if (word not in stoplist) and (word in word_vectors)]

    print(words)

    for word in words:

        vec += word_vectors[word]

    return vec
from sklearn.metrics.pairwise import cosine_similarity
s1 = train_data.iloc[5]['question1']

s2 = train_data.iloc[5]['question2']                  



wv1 = get_vector_for_sentence(s1)

wv2 = get_vector_for_sentence(s2)
print(cosine_similarity(wv1.reshape(1,-1),wv2.reshape(1,-1)))
train_data.head(10)[['question1','question2']].apply(lambda x: get_vector_for_sentence(x),axis=0)