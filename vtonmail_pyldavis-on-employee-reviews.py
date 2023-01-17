# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



import spacy

nlp = spacy.load('en')



from spacy.tokenizer import Tokenizer

tokenizer = Tokenizer(nlp.vocab)



import nltk

nltk.download('wordnet')



from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords



import gensim

from gensim.parsing.preprocessing import strip_punctuation

from gensim.parsing.preprocessing import remove_stopwords



from gensim.models import Phrases

from gensim.models.phrases import Phraser

from gensim import models



import pyLDAvis.gensim



import warnings

warnings.filterwarnings("ignore")



def show_pyldavis(docs, num_topics):



#docs is a list of strings

#num_topics for the LDA model



  #docs is the list of documents (list of strings)



  docs = [remove_stopwords(doc.lower()) for doc in docs]



  token_ = [strip_punctuation(' '.join([str(x) for x in nlp(doc)])) for doc in docs]



  token_ = [x.split(" ") for x in token_ if len(x)>2]



  lmtzr = WordNetLemmatizer()



  for token in token_:

      token = [lmtzr.lemmatize(x) for x in token if len(x.strip())>2]

      token = [x for x in token if x not in set(stopwords.words('english'))]



  bigram = Phrases(token_, min_count=5, threshold=2,delimiter=b' ')



  bigram_phraser = Phraser(bigram)



  bigram_token = []

  for sent in token_:

      bigram_token.append(bigram_phraser[sent])



  #now you can make dictionary of bigram token 

  dictionary = gensim.corpora.Dictionary(bigram_token)



  corpus = [dictionary.doc2bow(text) for text in bigram_token]



  lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=20)



  viz = pyLDAvis.gensim.prepare(lda, corpus, dictionary)



  return pyLDAvis.display(viz)
df = pd.read_csv('../input/employee_reviews.csv')
%%time

show_pyldavis(list(df[df['company']=='google']['pros'].values)+list(df[df['company']=='google']['cons'].values), 15)