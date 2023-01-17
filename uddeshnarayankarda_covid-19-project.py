import numpy as np 
import pandas as pd 
import json
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import os
from gensim import corpora, models
from tqdm import tqdm
import pandas as pd

import gc      
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from pprint import pprint

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','fig','figure','copyright'])

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

count = 0
all_text = []
punctuations="?:!.,;(){}[]"
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.split(".")[-1] == "json":
            if count == 10000:
                break
            f = open(os.path.join(dirname, filename))
            data = json.load(f)
            f.close()
            body_text = ''.join(k['text'] for k in data['body_text'])
            if 'abstract' in data.keys():
                abstract_text = ''.join(k['text'] for k in data['abstract'])
            else:
                abstract_text = ''
            text = '' + abstract_text + body_text
            if 'corona' in text.lower() or 'covid' in text.lower() and 'medical care' in text.lower():
                word_tokens = word_tokenize(text)
                filtered_sentence = []
                
                for w in word_tokens: 
                    if w not in stop_words: 
                        filtered_sentence.append(w) 

                remove_punc = []
                for word in filtered_sentence:
                    if word not in punctuations and len(word) > 3:
                        remove_punc.append(word.lower())
                
                lemmetized = []        
                for word in remove_punc:
                    lemmetized.append(wordnet_lemmatizer.lemmatize(word))
                    
                all_text.append(lemmetized)
            count += 1
print(len(all_text))

bigram = gensim.models.Phrases(all_text, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[all_text], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

data_words_bigrams = make_bigrams(all_text)

id2word = corpora.Dictionary(data_words_bigrams)

texts = data_words_bigrams

corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
