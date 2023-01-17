# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

import nltk

import multiprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# I load all the data to the RAM because the data is not that big,

# but in the general case you should consider defining a class for stream sentences

def read_text(filepath):

    def tokenize(text):

        word_num = 0

        sentences = []

        for sent in nltk.sent_tokenize(text, language='english'):

            tokens = []

            for word in nltk.word_tokenize(sent, language='english'):

                if len(word) < 2 or word in nltk.corpus.stopwords.words('english'):

                    continue

                tokens.append(word.lower())

            if tokens: 

                sentences.append(tokens)

                word_num += len(tokens)

        return sentences, word_num

    

    sentences = []

    with open(filepath) as inp:

        text, word_num = tokenize(inp.read())#.replace("\n", " ").replace("\t", " ").replace(". ", ".")

        print("# words:", word_num)

        return text

        



text_medit = read_text("../input/pg2680.txt")

text_morm = read_text("../input/pg17.txt")

text_koran = read_text("../input/pg2800.txt")

text_buddha = read_text("../input/35895-0.txt")

text_king = read_text("../input/pg10.txt")
n_iterations = 300

n_workers = multiprocessing.cpu_count()

n_dim = 100



w2v_medit =   gensim.models.Word2Vec(text_medit, size=n_dim, sg=1, iter=n_iterations, workers=n_workers)

w2v_morm =     gensim.models.Word2Vec(text_morm, size=n_dim, sg=1, iter=n_iterations, workers=n_workers)

w2v_koran =   gensim.models.Word2Vec(text_koran, size=n_dim, sg=1, iter=n_iterations, workers=n_workers)

w2v_buddha = gensim.models.Word2Vec(text_buddha, size=n_dim, sg=1, iter=n_iterations, workers=n_workers)

w2v_king =     gensim.models.Word2Vec(text_king, size=n_dim, sg=1, iter=n_iterations, workers=n_workers)
def most_similar(pos_word, neg_word = []):

    

    def _print(w_list, text):

        print("Word+:", pos_word, "|", "Word-:", neg_word, "Text:", text, sep = "\t")

        for w in w_list:

            print(w)

        print()

            

    def _most_sim(w2v, pos_word, neg_word, text):    

        try:    

            _print(w2v.most_similar(pos_word, neg_word), text)

        except KeyError:

            print("Words ", pos_word, neg_word, "not in the text ", text, "\n")

            

    _most_sim(w2v_medit, pos_word, neg_word, "Meditations")

    _most_sim(w2v_morm, pos_word, neg_word, "Mormons")

    _most_sim(w2v_koran, pos_word, neg_word, "Koran")

    _most_sim(w2v_buddha, pos_word, neg_word, "Buddha")

    _most_sim(w2v_king, pos_word, neg_word, "King James' Bible")
most_similar("god")
most_similar("buddha")
most_similar("man")
most_similar("woman")
most_similar("love")
most_similar("death")
most_similar("life")
most_similar("soul")
# god - man + woman

most_similar(["god", "woman"], ["man"])