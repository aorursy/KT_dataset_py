# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
f = open(r'/kaggle/input/feedback_cs2012.txt', 'r')
text=f.read()
print(text)
import nltk
tokens = nltk.word_tokenize(text)
# print(tokens)
# nltk.pos_tag(tokens)
# from nltk.corpus import brown
# brown_tagged_sents = brown.tagged_sents(categories='news')
# print(brown_tagged_sents[0])
from nltk.corpus import treebank
print(len(treebank.tagged_sents()))

train_data = treebank.tagged_sents()[:3000]
# train_data = treebank.tagged_sents()

test_data = treebank.tagged_sents()[3000:]
unigram_tagger = nltk.UnigramTagger(train_data)
unigramTagged = unigram_tagger.tag(tokens)
f = open(r'UnigramTaggedPOS.txt', 'w')

for i in unigramTagged:
    f.write(str(i))
    f.write('\r\n')

f.close()

unigram_tagger.evaluate(test_data)
bigram_tagger = nltk.BigramTagger(train_data)
bigramTagged = bigram_tagger.tag(tokens)

f = open(r'BigramTaggedPOS.txt', 'w')

for i in bigramTagged:
    f.write(str(i))
    f.write('\r\n')

f.close()

bigram_tagger.evaluate(test_data)
from nltk.tag.perceptron import PerceptronTagger
pretrain = PerceptronTagger(train_data)
perceptronTagged = pretrain.tag(tokens)

f = open(r'perceptronTaggedPOS.txt', 'w')

for i in perceptronTagged:
    f.write(str(i))
    f.write('\r\n')

f.close()

pretrain.evaluate(test_data)
from nltk.tag import tnt
tnt_pos_tagger = tnt.TnT(N=100)
tnt_pos_tagger.train(train_data)

# tntTagged = tnt_pos_tagger.tag(tokens[:500])
tntTagged = tnt_pos_tagger.tag(tokens)

f = open(r'tntTaggedPOS.txt', 'w')

for i in tntTagged:
    f.write(str(i))
    f.write('\r\n')

f.close()

tnt_pos_tagger.evaluate(test_data)