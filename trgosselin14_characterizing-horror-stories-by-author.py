import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from collections import Counter

import string

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

train.head()
train['tokens'] = [nltk.word_tokenize(i) for i in train['text']]

train.head()
#Count of Punc.

punc_list = [string.punctuation]

def count_punc_in_tokens(series):

    counts = []

    for s in series:

        counter = 0

        for i in s:

            if i in punc_list[0]:

                counter += 1

        counts.append(counter)

    return counts

train['count of punc'] = count_punc_in_tokens(train['tokens'])
# Count of Non-Punc

punc_list = [string.punctuation]

def count_words_in_tokens(series):

    counts = []

    for s in series:

        counter = 0

        for i in s:

            if i not in punc_list[0]:

                counter += 1

        counts.append(counter)

    return counts

train['count of not punc'] = count_words_in_tokens(train['tokens'])

train['words_to_punc_ratio'] = train['count of not punc']/train['count of punc']

def tokens_without_punc(series):

    row = []

    for s in series:

        tokens = []

        for i in s:

            if i not in punc_list[0]:

                tokens.append(i)

        row.append(tokens)

    return row



train['token_without_punc'] = [i for i in tokens_without_punc(train['tokens'])]
def pos_used(series):

    counts = []

    itter = 0

    for s in series:

        cnt = Counter()

        tags = nltk.pos_tag(s)

        print(tags)

        pos_list = []

        for i in tags:

            pos_list.append(i[1])

        for i in pos_list:

            cnt[i] += 1

        count = list(cnt.items())

        df = pd.DataFrame(count,  columns=['POS', str(itter)])

        df.set_index('POS', inplace = True)

        counts.append(df)

        itter += 1

    #[i.set_index('POS', inplace = True) for i in counts]

    pos_df = pd.concat(counts, axis=1)

    return pos_df



pos_used = pos_used(train['token_without_punc'])
pos_used.fillna(0,inplace=True)

pos_used['SUM'] = pos_used.sum(axis=1)

pos_used = pos_used[pos_used['SUM'] >= pos_used['SUM'].quantile(.5)]

pos_used.drop(['SUM'], axis=1, inplace=True)
#print(pos_used.T.shape) # 19579, 38

#print(train.shape) # 19579, 8

pos_df = pos_used.transpose()

pos_df.reset_index(inplace = True)

pos_df.drop(['index'], inplace = True, axis= 1)
train2 = pd.concat([train,pos_df], axis = 1)
train = train2.drop(['id', 'text', 'tokens', 'token_without_punc','count of punc', 'count of not punc'], axis = 1)
train.head()
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.ensemble import RandomForestClassifier

y = train.iloc[:,0]

X = train.iloc[:,1:]

X = StandardScaler().fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .4, random_state = 42)
clf = OneVsRestClassifier(SVC(kernel = 'linear'))

clf.fit(X_train,y_train)

cross_val_score(clf, X_train, y_train, cv=3, scoring = 'accuracy')
clf = RandomForestClassifier()

clf.fit(X_train,y_train)

cross_val_score(clf, X_train, y_train, cv=3, scoring = 'accuracy')