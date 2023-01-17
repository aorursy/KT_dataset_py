# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from urllib.parse import urlparse # working with url parsing to find domains

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
articles = pd.read_csv('../input/articlesCSV.csv', 
                       names=['article_id', 'time', 'title', 'url', 'symbol', 'volume'],
                       index_col=0,
                       parse_dates=[1])
articles.head()
# Get the top 5 repeating urls and plot them...
articles.url.value_counts().head(n=5).plot(kind="bar", title="Outlier Duplicate URLs by duplicate count")
top10articleUrls = list(articles.url.value_counts().head(n=10).keys())
top10 = articles[articles.url.isin(top10articleUrls)][['url', 'symbol', 'time']].set_index('time', drop=True)
pd.Series(top10.sort_index().index.date).value_counts().plot.bar(title="Days we got duplicate results", figsize=(10,5))
# duplicate the url columns
urls = articles.url
articles = articles.assign(domains=urls)
# create a function to extract the domain from a given url
def extractDomain(url):
    url = str(url)
    if(url.startswith('http://') or url.startswith('https://')):
        return urlparse(url).netloc;
    return url
articles.domains = articles.domains.apply(extractDomain)
uniqueDomains = articles.domains.value_counts()
uniqueDomains20 = articles.domains.value_counts().head(20)
uniqueDomains20.plot(kind='pie', figsize=(15, 15), title="Top 20 Domains of Articles")
twentiethRecord = uniqueDomains20[19:]
frequencyCutoff = twentiethRecord[0]
frequencyCutoff # past this number, is the "other" portion of the bar chart
# group the non-twenty domains into "other"
lesserDomains = uniqueDomains[uniqueDomains < frequencyCutoff]
lesserDomains.head()
# compress the other domains into 'other'
otherCount = lesserDomains.sum()
otherCount
top20Domains = uniqueDomains[uniqueDomains >= frequencyCutoff]
top20DomainsPlus = pd.Series.to_frame(top20Domains)

top20DomainsPlus.loc['other'] = otherCount
top20DomainsPlus.plot(kind='pie', y='domains', figsize=(10,10), title="Top 20 Article Domains and Other")
# 50 unique article titles...
top50ArticleTitles = articles.title.value_counts().head(50);
articlesWithTop50Titles = articles[articles.title.isin(top50ArticleTitles.keys())]
articlesWithTop50Titles.title.count()
# ... 892 Articles
# 119 symbols
articlesWithTop50Titles.symbol.value_counts().count()
# load the scored articles

articlesScored = pd.read_csv('../input/spamOrHamArticles.csv')
# probably easier to work with 1 for [s]pam and 0 for [h]am.
# ty: https://stackoverflow.com/questions/23307301/pandas-replacing-column-values-in-dataframe
articlesScored['spam'] = articlesScored['spam'].map({'s': 1, 'h': 0})
articlesScored.head()
# tokenize all the words, only pay attention to main words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#TODO: nice, one pass function instead of iterating three times
#def tokenizeAndClean(strs):
    

sw = stopwords.words('english')
# tokenize and remove stop words
articlesScored['tokens'] = articlesScored['title'].map(word_tokenize)
articlesScored['tokens'] = articlesScored['tokens'].map(lambda x: list(map(str.lower, x)))
articlesScored['tokens'] = articlesScored['tokens'].apply(lambda x: [w for w in x if w not in sw])
articlesScored.head()
from sklearn.model_selection import train_test_split

train, test = train_test_split(articlesScored, test_size=0.3)
# find unique words in the entire df.
wordFreq = pd.DataFrame(columns=['spam', 'ham', 'spamMissing', 'hamMissing'])

totalSpam = train.spam.sum()
totalHam = train.spam.count() - totalSpam
for i, r in train.iterrows():
    isSpamRow = r.spam == 1
    tokens = r.tokens
    for t in tokens:
        # if the word's not in the frequency list, add a spot for it
        indicies = wordFreq.index.get_values()
        if t not in indicies:
            wordFreq.loc[t] = [0, 0, totalSpam, totalHam]
        if isSpamRow:
            wordFreq.loc[t].spam += 1
            wordFreq.loc[t].spamMissing -= 1
        else:
            wordFreq.loc[t].ham += 1
            wordFreq.loc[t].hamMissing -=1

wordFreq
# probability is spam
wordFreq['spamProb'] = (wordFreq.spam) / (wordFreq.spam.sum() - wordFreq.spam)
wordFreq['hamProb'] = (wordFreq.ham) / (wordFreq.ham.sum() - wordFreq.ham)

wordFreq
# Testing

numCorrect = 0
thresh = 0.3 # we want to be 80% sure it's ham
   
for i, r in test.iterrows():
    ts = r.tokens
    probSp = 1
    probHam = 1
    for ind, row in wordFreq.iterrows():
        if row.name in ts:
            probSp *= row.spamProb
            probHam *= row.hamProb
        else:
            probSp *= (1 - row.spamProb)
            probHam *= (1 - row.hamProb)
    if probSp + probHam == 0:
        shouldKeepAsHam = 0
    else:
        shouldKeepAsHam = probHam / (probSp + probHam)
    if shouldKeepAsHam >= thresh:
        # classified as ham
        if row.spam == 0:
            numCorrect += 1
    else:
        # classified as spam
        if row.spam == 1:
            numCorrect += 1


accuracy = numCorrect / test.title.count()
print(f'Accuracy: {accuracy}')
