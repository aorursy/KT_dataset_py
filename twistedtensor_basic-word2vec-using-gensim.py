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
import nltk

import gensim

import multiprocessing
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
# Remove all non-letter characters and make everything lowercase.

train['comment_text'] = train['comment_text'].str.replace('[^a-zA-Z]',' ').str.lower()

test['comment_text'] = test['comment_text'].str.replace('[^a-zA-Z]',' ').str.lower()
train['comment_text'].head(10)
# Remove stop words with regex. '\\b' matches any break (space or linebreak or whatever) and '|'

# is an or operator. So, for example '\\ba\\b|\\bis\\b|\\band\\b' will match 'a', 'is' or 'and'.

stop_re = '\\b'+'\\b|\\b'.join(nltk.corpus.stopwords.words('english'))+'\\b'

train['comment_text'] = train['comment_text'].str.replace(stop_re, '')

test['comment_text'] = test['comment_text'].str.replace(stop_re, '')



train['comment_text'].head(10)
# Tokenize words

train['comment_text'] = train['comment_text'].str.split()

test['comment_text'] = test['comment_text'].str.split()



train['comment_text'].head(10)
# Detect common phrases so that we may treat each one as its own word

phrases = gensim.models.phrases.Phrases(train['comment_text'].tolist())

phraser = gensim.models.phrases.Phraser(phrases)

train_phrased = phraser[train['comment_text'].tolist()]
# Gensim has support for multi-core systems

multiprocessing.cpu_count()
# I have no reason in mind to change the default word2vec parameters, so I will use the defaults

w2v = gensim.models.word2vec.Word2Vec(sentences=train_phrased,workers=32)
w2v.save('w2v_v1')