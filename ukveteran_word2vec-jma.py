import re
import numpy as np

from gensim.models import Word2Vec
from nltk.corpus import gutenberg
from multiprocessing import Pool
from scipy import spatial
sentences = list(gutenberg.sents('shakespeare-hamlet.txt'))   # import the corpus and convert into a list
print('Type of corpus: ', type(sentences))
print('Length of corpus: ', len(sentences))
print(sentences[0])    # title, author, and year
print(sentences[1])
print(sentences[10])
for i in range(len(sentences)):
    sentences[i] = [word.lower() for word in sentences[i] if re.match('^[a-zA-Z]+', word)]
print(sentences[0])    # title, author, and year
print(sentences[1])
print(sentences[10])
model = Word2Vec(sentences = sentences, size = 100, sg = 1, window = 3, min_count = 1, iter = 10, workers = Pool()._processes)
model.init_sims(replace = True)
model.save('word2vec_model')
model = Word2Vec.load('word2vec_model')
model.most_similar('hamlet')
v1 = model['king']
v2 = model['queen']
# define a function that computes cosine similarity between two words
def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)
cosine_similarity(v1, v2)