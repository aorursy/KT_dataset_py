import re
import numpy as np

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
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
for i in range(len(sentences)):
    sentences[i] = TaggedDocument(words = sentences[i], tags = ['sent{}'.format(i)])    # converting each sentence into a TaggedDocument
sentences[0]
model = Doc2Vec(documents = sentences, dm = 1, size = 100, window = 3, min_count = 1, iter = 10, workers = Pool()._processes)
model.init_sims(replace = True)
model.save('doc2vec_model')
model = Doc2Vec.load('doc2vec_model')