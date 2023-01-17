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
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from scipy.special import logit, expit



import copy

import re

from keras.preprocessing.text import text_to_word_sequence

from nltk import WordNetLemmatizer



class BaseTokenizer(object):

    def process_text(self, text):

        raise NotImplemented



    def process(self, texts):

        for text in texts:

            yield self.process_text(text)





RE_PATTERNS = {

    ' american ':

        [

            'amerikan'

        ],



    ' adolf ':

        [

            'adolf'

        ],





    ' hitler ':

        [

            'hitler'

        ],



    ' fuck':

        [

            '(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',

            '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',

            ' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k', 'f u u c',

            '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*',

            'feck ', ' fux ', 'f\*\*', 

            'f\-ing', 'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck'

        ],



    ' ass ':

        [

            '[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\$\$'

                                                           '[^a-z]anus', ' a\*s\*s', '[^a-z]ass[^a-z ]',

            'a[@#\$%\^&\*][@#\$%\^&\*]', '[^a-z]anal ', 'a s s'

        ],



    ' ass hole ':

        [

            ' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\$\$hole'

        ],



    ' bitch ':

        [

            'b[w]*i[t]*ch', 'b!tch',

            'bi\+ch', 'b!\+ch', '(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)',

            'biatch', 'bi\*\*h', 'bytch', 'b i t c h'

        ],



    ' bastard ':

        [

            'ba[s|z]+t[e|a]+rd'

        ],



    ' trans gender':

        [

            'transgender'

        ],



    ' gay ':

        [

            'gay'

        ],



    ' cock ':

        [

            '[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',

            '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'

        ],



    ' dick ':

        [

            ' dick[^aeiou]', 'deek', 'd i c k'

        ],



    ' suck ':

        [

            'sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'

        ],



    ' cunt ':

        [

            'cunt', 'c u n t'

        ],



    ' bull shit ':

        [

            'bullsh\*t', 'bull\$hit'

        ],



    ' homo sex ual':

        [

            'homosexual'

        ],



    ' jerk ':

        [

            'jerk'

        ],



    ' idiot ':

        [

            'i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots'

                                                                                      'i d i o t'

        ],



    ' dumb ':

        [

            '(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'

        ],



    ' shit ':

        [

            'shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\$hit', 's h i t'

        ],



    ' shit hole ':

        [

            'shythole'

        ],



    ' retard ':

        [

            'returd', 'retad', 'retard', 'wiktard', 'wikitud'

        ],



    ' rape ':

        [

            ' raped'

        ],



    ' dumb ass':

        [

            'dumbass', 'dubass'

        ],



    ' ass head':

        [

            'butthead'

        ],



    ' sex ':

        [

            'sexy', 's3x', 'sexuality'

        ],





    ' nigger ':

        [

            'nigger', 'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'

        ],



    ' shut the fuck up':

        [

            'stfu'

        ],



    ' pussy ':

        [

            'pussy[^c]', 'pusy', 'pussi[^l]', 'pusses'

        ],



    ' faggot ':

        [

            'faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',

            '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot',

        ],



    ' mother fucker':

        [

            ' motha ', ' motha f', ' mother f', 'motherucker',

        ],



    ' whore ':

        [

            'wh\*\*\*', 'w h o r e'

        ],

}





class PatternTokenizer(BaseTokenizer):

    def __init__(self, lower=True, initial_filters=r"[^a-z0-9!@#\$%\^\&\*_\-,\.' ]", patterns=RE_PATTERNS,

                 remove_repetitions=True):

        self.lower = lower

        self.patterns = patterns

        self.initial_filters = initial_filters

        self.remove_repetitions = remove_repetitions



    def process_text(self, text):

        x = self._preprocess(text)

        for target, patterns in self.patterns.items():

            for pat in patterns:

                x = re.sub(pat, target, x)

        x = re.sub(r"[^a-z' ]", ' ', x)

        return x.split()



    def process_ds(self, ds):

        ### ds = Data series



        # lower

        ds = copy.deepcopy(ds)

        if self.lower:

            ds = ds.str.lower()

        # remove special chars

        if self.initial_filters is not None:

            ds = ds.str.replace(self.initial_filters, ' ')

        # fuuuuck => fuck

        if self.remove_repetitions:

            pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 

            ds = ds.str.replace(pattern, r"\1")



        for target, patterns in self.patterns.items():

            for pat in patterns:

                ds = ds.str.replace(pat, target)



        ds = ds.str.replace(r"[^a-z' ]", ' ')



        return ds.str.split()



    def _preprocess(self, text):

        # lower

        if self.lower:

            text = text.lower()



        # remove special chars

        if self.initial_filters is not None:

            text = re.sub(self.initial_filters, ' ', text)



        # fuuuuck => fuck

        if self.remove_repetitions:

            pattern = re.compile(r"(.)\1{2,}", re.DOTALL)

            text = pattern.sub(r"\1", text)

        return text
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



tokenizer = PatternTokenizer()

train_text = tokenizer.process_ds(train["comment_text"]).str.join(sep=" ")

test_text = tokenizer.process_ds(test["comment_text"]).str.join(sep=" ")



all_text = pd.concat([train_text, test_text])
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



word_vectorizer = CountVectorizer(stop_words = 'english',analyzer='word')

word_vectorizer.fit(all_text)

train_features = word_vectorizer.transform(train_text)

test_features = word_vectorizer.transform(test_text)
losses = []

predictions = {'id': test['id']}

for class_name in classes:

    train_target = train[class_name]

    classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=10, max_features="sqrt", max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=3, min_samples_split=8,

            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,

            oob_score=True, random_state=None, verbose=1,

            warm_start=False)

    

    results = cross_val_score(classifier, train_features, train_target, cv=3, scoring='f1_micro')

    cv_loss = results.mean()

    losses.append(cv_loss)

    print('CV score for class {} is {}'.format(class_name, cv_loss))

    print("CV accuracy score: {:.2f}%".format(results.mean()*100))

    

    classifier.fit(train_features, train_target)

    predictions[class_name] = expit(logit(classifier.predict_proba(test_features)[:, 1]))



print('Total CV score is {}'.format(np.mean(losses)))