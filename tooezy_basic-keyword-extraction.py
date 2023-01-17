import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk as nlp

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/papers.csv")
data.head()
print(data.paper_text[1][:500], "...")

print("\ntotal length", len(data.paper_text[1]))
data.info()
sum(data.abstract == "Abstract Missing")
import re

data.paper_text = data.paper_text.apply(lambda x: re.sub("(\W)", " ", x))
tokenizer = nlp.WordPunctTokenizer()

data["word_count"] = data.paper_text.apply(lambda x: len(tokenizer.tokenize(x)))

data[["word_count", "paper_text"]].head()
freq = pd.Series(" ".join(data.paper_text).split()).value_counts()

print(freq.head(10))

print(freq.tail(10))
lemma = nlp.WordNetLemmatizer()

data.paper_text = data.paper_text.apply(lambda x: lemma.lemmatize(x))
data.paper_text = data.paper_text.apply(lambda x: x.lower())
from wordcloud import WordCloud, STOPWORDS

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

stopword_list = set(stopwords.words("english"))



word_cloud = WordCloud(

                          background_color='white',

                          stopwords=stopword_list,

                          max_words=100,

                          max_font_size=50, 

                          random_state=42

                         ).generate(str(data.paper_text))

print(word_cloud)

fig = plt.figure(1)

plt.imshow(word_cloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=900)
from sklearn.feature_extraction.text import TfidfVectorizer



tf_idf = TfidfVectorizer(max_df=0.8,stop_words=stopword_list, max_features=10000, ngram_range=(1,3))

tf_idf.fit(data.paper_text)
doc = pd.Series(data.paper_text[500])

doc_vector = tf_idf.transform(doc)
#Function for sorting tf_idf in descending order

from scipy.sparse import coo_matrix

def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

 

def extract_topn_from_vector(feature_names, sorted_items, topn=10):

    """get the feature names and tf-idf score of top n items"""

    

    #use only topn items from vector

    sorted_items = sorted_items[:topn]

    score_vals = []

    feature_vals = []

    

    # word index and corresponding tf-idf score

    for idx, score in sorted_items:

        

        #keep track of feature name and its corresponding score

        score_vals.append(round(score, 3))

        feature_vals.append(feature_names[idx])



    #create a tuples of feature,score

    #results = zip(feature_vals,score_vals)

    results= {}

    for idx in range(len(feature_vals)):

        results[feature_vals[idx]]=score_vals[idx]

    

    return results

#sort the tf-idf vectors by descending order of scores

sorted_items=sort_coo(doc_vector.tocoo())

#extract only the top n; n here is 10

feature_names = tf_idf.get_feature_names()

keywords=extract_topn_from_vector(feature_names,sorted_items,5)

 

# now print the results

print("\nAbstract:")

print(doc[0][:1000])
print("Keywords:")

for k in keywords:

    print(k,keywords[k])