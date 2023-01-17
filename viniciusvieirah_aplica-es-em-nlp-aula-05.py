# Francis Pimentel

# Vinicius Vieira
import nltk

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string



hamlet_raw = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')

sentences = sent_tokenize(hamlet_raw)



all_words = [nltk.word_tokenize(sent) for sent in sentences]



for i in range(len(all_words)):  

    all_words[i] = [w.lower() for w in all_words[i] if w not in stopwords.words('english') and w not in string.punctuation]

    

print(all_words[:10])
hamlet_raw
from gensim.models import Word2Vec



#sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.

word2vec = Word2Vec(all_words, min_count=2, sg=1)



vocabulary = word2vec.wv.vocab  

print(vocabulary)
v1 = word2vec.wv['heart']

print(v1)
sim_words = word2vec.wv.most_similar('truth')

print(sim_words)
!pip install glove_python
from glove import Corpus, Glove





corpus = Corpus() 



corpus.fit(all_words, window=10)



#creating a Glove object which will use the matrix created in the above lines to create embeddings

#We can set the learning rate as it uses Gradient Descent and number of components



glove = Glove(no_components=5, learning_rate=0.05)

 

glove.fit(corpus.matrix, epochs=30, no_threads=1, verbose=True)

glove.add_dictionary(corpus.dictionary)

glove.save('glove.model')



glove.add_dictionary(corpus.dictionary)
print(corpus.dictionary)
print(corpus.matrix)
# Vamos ver os vetores de uma palavra:

print(glove.word_vectors[glove.dictionary['heart']])

glove.most_similar('truth')
from gensim.models import Word2Vec



#sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.





class Word2VecTransformer(object):

    

    ALGO_SKIP_GRAM=1

    ALGO_CBOW=2    

    

    def __init__(self, algo=1):

        self.algo = algo

    

    def fit(self, X, y=None):     

        X = [nltk.word_tokenize(x) for x in X]

        

        self.word2vec = Word2Vec(X, min_count=2, sg=self.algo)

        

        # Pegamos a dimensão da primeira palavra, para saber quantas dimensões estamos trabalhando,

        # assim podemos ajustar nos casos em que aparecerem palavras que não existirem no vocabulário.

        first_word = next(iter(self.word2vec.wv.vocab.keys()))

        self.num_dim = len(self.word2vec[first_word])       

        

        return self

    

    def transform(self, X, Y=None):        

        X = [nltk.word_tokenize(x) for x in X]

        

        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.num_dim)], axis=0) 

                         for words in X])



        

    def get_params(self, deep=True):

        return {}

import nltk



nltk.download('nps_chat')

from nltk.corpus import nps_chat



print(nps_chat.fileids())
import nltk



x_data_nps = []



for fileid in nltk.corpus.nps_chat.fileids():

    x_data_nps.extend([post.text for post in nps_chat.xml_posts(fileid)])



y_data_nps = [0] * len(x_data_nps)



x_data_gut = []

for fileid in nltk.corpus.gutenberg.fileids():

    x_data_gut.extend([' '.join(sent) for sent in nltk.corpus.gutenberg.sents(fileid)])

    

y_data_gut = [1] * len(x_data_gut)



x_data_full = x_data_nps[:500] + x_data_gut[:500]

print(len(x_data_full))

y_data_full = y_data_nps[:500] + y_data_gut[:500]

print(len(y_data_full))
x_data_full
words =''.join(x_data_full)
sentences = sent_tokenize(words)
all_words = [nltk.word_tokenize(sent) for sent in sentences]



for i in range(len(all_words)):  

    all_words[i] = [w.lower() for w in all_words[i] if w not in stopwords.words('english') and w not in string.punctuation]

    

print(all_words[:10])
# Skip-Gram

#sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.

word2vec = Word2Vec(all_words, min_count=2, sg=1)



vocabulary = word2vec.wv.vocab  

print(vocabulary)

v1 = word2vec.wv['hell']

print(v1)
sim_words = word2vec.wv.most_similar('hell')

print(sim_words)
#CBOW

#sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.

word2vec = Word2Vec(all_words, min_count=2, sg=0)



vocabulary = word2vec.wv.vocab  

print(vocabulary)
v1 = word2vec.wv['hell']

print(v1)
sim_words = word2vec.wv.most_similar('hell')

print(sim_words)
#GloVe
corpus = Corpus() 



corpus.fit(all_words, window=10)



#creating a Glove object which will use the matrix created in the above lines to create embeddings

#We can set the learning rate as it uses Gradient Descent and number of components



glove = Glove(no_components=5, learning_rate=0.05)

 

glove.fit(corpus.matrix, epochs=30, no_threads=1, verbose=True)

glove.add_dictionary(corpus.dictionary)

glove.save('glove.model')



glove.add_dictionary(corpus.dictionary)
print(corpus.dictionary)
print(corpus.matrix)
# Vamos ver os vetores de uma palavra:

print(glove.word_vectors[glove.dictionary['hell']])

glove.most_similar('hell')