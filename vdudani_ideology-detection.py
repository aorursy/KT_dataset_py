# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import operator

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))
from nltk import PorterStemmer
# Any results you write to the current directory are saved as output.
TRAINING_DATA_PATH = '../input/train.csv'


def getSortedWordFrequencies(doc_list, topNWords):
    count_vect = CountVectorizer(input='content')
    matrix = count_vect.fit_transform(doc_list)
    frequencyCounts = np.squeeze(np.asarray(matrix.sum(axis=0)))
    frequencies = list(zip(count_vect.get_feature_names(), frequencyCounts))
    frequencies.sort(key=lambda x: x[1], reverse=True)
    return frequencies[:topNWords]


data = pd.read_csv(TRAINING_DATA_PATH)
sentences = data['sentence']

wordFrequencies = getSortedWordFrequencies(sentences, 100)

print("Word Frequencies")
for word, frequency in wordFrequencies:
    print('{:s} : {:d}'.format(word, frequency))
np.random.seed(23)

train = pd.read_csv('../input/train.csv', index_col=None)
test = pd.read_csv('../input/test.csv', index_col=None)

def makeTaxWords():
    porterStemmer = PorterStemmer()
    taxWords = [
        'tax',
        'tariff',
        'levy',
        'fine',
        'economy',
        'money',
    ]
    return list(map(porterStemmer.stem, taxWords))


def hasTaxWord(sentence):
    words = sentence.split(' ')
    for word in words:
        if word in taxWords:
            return True
    return False


taxWords = makeTaxWords()
trainTax = train[train.apply(lambda x: hasTaxWord(x['sentence']), axis=1)]
testTax = test[test.apply(lambda x: hasTaxWord(x['sentence']), axis=1)]

trainTax.to_csv('trainTax.csv', index=False)
testTax.to_csv('testTax.csv', index=False)

def makeDrugsWords():
    porterStemmer = PorterStemmer()
    drugsWords = [
        'marijuana',
        'pot',
        'dope',
        'drug',
        'opioid',
        'heroin',
        'cocaine',
        'cocaine',
    ]
    return list(map(porterStemmer.stem, drugsWords))


def hasDrugsWord(sentence):
    words = sentence.split(' ')
    for word in words:
        if word in drugsWords:
            return True
    return False


drugsWords = makeDrugsWords()
trainDrugs = train[train.apply(lambda x: hasDrugsWord(x['sentence']), axis=1)]
testDrugs = test[test.apply(lambda x: hasDrugsWord(x['sentence']), axis=1)]

trainDrugs.to_csv('trainDrugs.csv', index=False)
testDrugs.to_csv('testDrugs.csv', index=False)



trainModern = train[train['year'] > 2010]
testModern = test[test['year'] > 2010]

trainModern.to_csv('trainModern.csv', index=False)
testModern.to_csv('testModern.csv', index=False)



def makePoliticalWords():
    porterStemmer = PorterStemmer()
    politicalWords = [
        'political',
        'tariff',
        'levy',
        'fine',
        'economy',
        'money',
        'bipartisan',
        'big government',
        'bleeding heart',
        'checks and balances',
        'gerrymander',
        'left wing',
        'right wing',
        'liberal',
        'conservative',
        'witch hunt',
        'abortion',
        'gay',
        'homosexual',
        'marijuana',
        'teabagger',
        'ground zero',
        'elite',
        'climate change',
        'global warming',
        'job',
        'recession',
        'gun',
        'right',
        'amendment',
        'security',
        'army',
        'armed forces',
        'capitalism',
        'free market',
        'socialist',
        'education',
        'student',
        'college',
        'teacher',
        'class',
        'responsibility',
        'free',
        'greed',
        'immigration',
        'life',
        'deal',
        'patriot',
        'peace',
        'welfare',
        'hand out',
        'handout',
        'rich',
        'science',
        'poor',
        'sustainable',
        'tolerance',
        'victim',
        'god',
        'christian',
        'faith',
        'pray',
        'religion',
        'prison',
        'criminal',
    ]
    return list(map(porterStemmer.stem, politicalWords))


def hasPoliticalWord(sentence):
    words = sentence.split(' ')
    for word in words:
        if word in politicalWords:
            return True
    return False


politicalWords = makePoliticalWords()
trainPolitical = train[train.apply(lambda x: hasPoliticalWord(x['sentence']), axis=1)]
testPolitical = test[test.apply(lambda x: hasPoliticalWord(x['sentence']), axis=1)]

trainPolitical.to_csv('trainPolitical.csv', index=False)
testPolitical.to_csv('testPolitical.csv', index=False)