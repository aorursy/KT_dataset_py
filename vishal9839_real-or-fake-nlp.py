# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
        import re # for regular expressions

import pandas as pd 

pd.set_option("display.max_colwidth", 200)

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

import string

import nltk # for text manipulation

import warnings

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
##Remove Pattern with username @

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

        

    return input_txt
##Remove all the hyperlinks from the texts

train['text'] = train['text'].str.replace('http\S+|www.\S+', '', case=False)

##Removing stopwords

# extracting the stopwords from nltk library

sw = stopwords.words('english')

# displaying the stopwords

np.array(sw);
def stopwords(text):

    '''a function for removing the stopword'''

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # joining the list of words with space separator

    return " ".join(text)
train['text'] = train['text'].apply(stopwords)
from sklearn import preprocessing



lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(train.target.values)
from sklearn.model_selection import train_test_split



xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, 

                                                  stratify=y, 

                                                  random_state=42, 

                                                  test_size=0.1, shuffle=True)
from sklearn.feature_extraction.text import TfidfVectorizer



tfv = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')
tfv.fit(list(xtrain) + list(xvalid))

xtrain_tfv =  tfv.transform(xtrain) 

xvalid_tfv = tfv.transform(xvalid)
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(C=1.0)

clf.fit(xtrain_tfv,ytrain)
from sklearn.metrics import f1_score

pred = clf.predict(xvalid_tfv)



f1score = f1_score(yvalid, pred)

print(f"Model Score: {f1score * 100} %")
target = clf.predict(tfv.transform(test.text.values))
sub['target'] = target

sub.to_csv("submission.csv", index=False)

sub.head()