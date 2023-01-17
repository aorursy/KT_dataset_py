import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/mbti-type/mbti_1.csv')
def get1(x):

    if x[0:1] == 'I':

        return 1

    else:

        return 0

def get2(x):

    if x[1:2] == 'N':

        return 1

    else:

        return 0

def get3(x):

    if x[2:3] == 'T':

        return 1

    else:

        return 0

def get4(x):

    if x[3:4] == 'J':

        return 1

    else:

        return 0

data['I'] = data['type'].apply(get1)

data['N'] = data['type'].apply(get2)

data['T'] = data['type'].apply(get3)

data['J'] = data['type'].apply(get4)
data.head()
from sklearn.utils import shuffle

X = data['posts']

y = data['I']

smallest_val = 0

if y.value_counts()[0] < y.value_counts()[1]:

    small = y.value_counts()[0]

    big = y.value_counts()[1]

    s_data = shuffle(data[data['I']==0][['posts','I']])

    b_data = shuffle(data[data['I']==1][['posts','I']])

else:

    small = y.value_counts()[1]

    big = y.values_counts()[0]

    s_data = shuffle(data[data['I']==1][['posts','I']])

    b_data = shuffle(data[data['I']==0][['posts','I']])

b_data = shuffle(b_data[:small])

for i in range(999):

    i_data = shuffle(pd.concat([b_data,s_data]))
col = 'N'



X = data['posts']

y = data[col]

smallest_val = 0

if y.value_counts()[0] < y.value_counts()[1]:

    small = y.value_counts()[0]

    big = y.value_counts()[1]

    s_data = shuffle(data[data[col]==0][['posts',col]])

    b_data = shuffle(data[data[col]==1][['posts',col]])

else:

    small = y.value_counts()[1]

    big = y.values_counts()[0]

    s_data = shuffle(data[data[col]==1][['posts',col]])

    b_data = shuffle(data[data[col]==0][['posts',col]])

b_data = shuffle(b_data[:small])

for i in range(999):

    n_data = shuffle(pd.concat([b_data,s_data]))
col = 'T'



X = data['posts']

y = data[col]

smallest_val = 0

if y.value_counts()[0] < y.value_counts()[1]:

    small = y.value_counts()[0]

    big = y.value_counts()[1]

    s_data = shuffle(data[data[col]==0][['posts',col]])

    b_data = shuffle(data[data[col]==1][['posts',col]])

else:

    small = y.value_counts()[1]

    big = y.value_counts()[0]

    s_data = shuffle(data[data[col]==1][['posts',col]])

    b_data = shuffle(data[data[col]==0][['posts',col]])

b_data = shuffle(b_data[:small])

for i in range(999):

    t_data = shuffle(pd.concat([b_data,s_data]))
col = 'J'



X = data['posts']

y = data[col]

smallest_val = 0

if y.value_counts()[0] < y.value_counts()[1]:

    small = y.value_counts()[0]

    big = y.value_counts()[1]

    s_data = shuffle(data[data[col]==0][['posts',col]])

    b_data = shuffle(data[data[col]==1][['posts',col]])

else:

    small = y.value_counts()[1]

    big = y.value_counts()[0]

    s_data = shuffle(data[data[col]==1][['posts',col]])

    b_data = shuffle(data[data[col]==0][['posts',col]])

b_data = shuffle(b_data[:small])

for i in range(999):

    j_data = shuffle(pd.concat([b_data,s_data]))
i_data.head()
n_data.head()
t_data.head()
j_data.head()
i_data['I'].value_counts()
n_data['N'].value_counts()
t_data['T'].value_counts()
j_data['J'].value_counts()
data['I'].value_counts()
i_data = data[['I','posts']]

n_data = data[['N','posts']]

t_data = data[['T','posts']]

j_data = data[['J','posts']]
import random

import progressbar as pb

for i in pb.progressbar(range(4680)):

    i_data = i_data.append(data[data['I']==0][['I','posts']].iloc[random.randint(0,len(data[data['I']==0][['I','posts']])-1)])

for i in pb.progressbar(range(6281)):

    n_data = n_data.append(data[data['N']==0][['N','posts']].iloc[random.randint(0,len(data[data['N']==0][['N','posts']])-1)])

for i in pb.progressbar(range(713)):

    t_data = t_data.append(data[data['T']==1][['T','posts']].iloc[random.randint(0,len(data[data['T']==1][['T','posts']])-1)])

for i in pb.progressbar(range(1807)):

    j_data = j_data.append(data[data['J']==1][['J','posts']].iloc[random.randint(0,len(data[data['J']==1][['J','posts']])-1)])
print(i_data['I'].value_counts())

print(n_data['N'].value_counts())

print(t_data['T'].value_counts())

print(j_data['J'].value_counts())
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
from bs4 import BeautifulSoup

import re

def cleanText(text):

    text = BeautifulSoup(text, "lxml").text

    text = re.sub(r'\|\|\|', r' ', text) 

    text = re.sub(r'http\S+', r'<URL>', text)

    return text

i_data['cleaned'] = i_data['posts'].apply(cleanText)

n_data['cleaned'] = n_data['posts'].apply(cleanText)

t_data['cleaned'] = t_data['posts'].apply(cleanText)

j_data['cleaned'] = j_data['posts'].apply(cleanText)
i_data = i_data.reset_index().drop('index',axis=1)

n_data = n_data.reset_index().drop('index',axis=1)

t_data = t_data.reset_index().drop('index',axis=1)

j_data = j_data.reset_index().drop('index',axis=1)
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

np.random.seed(1)



scoring = {'acc': 'accuracy','neg_log_loss': 'neg_log_loss','f1_micro': 'f1_micro'}

tfidf = CountVectorizer(ngram_range=(1, 1),stop_words='english',lowercase = True,max_features = 5000)
i_model = Pipeline([('tfidf', tfidf), ('model', DecisionTreeClassifier())])

n_model = Pipeline([('tfidf', tfidf), ('model', DecisionTreeClassifier())])

t_model = Pipeline([('tfidf', tfidf), ('model', DecisionTreeClassifier())])

j_model = Pipeline([('tfidf', tfidf), ('model', DecisionTreeClassifier())])

i_model.fit(i_data['cleaned'],i_data['I'])

n_model.fit(n_data['cleaned'],n_data['N'])

t_model.fit(t_data['cleaned'],t_data['T'])

j_model.fit(j_data['cleaned'],j_data['J'])
print("- - - RESULTS")

print("I Component Model Performance")

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

results_nb = cross_validate(i_model, i_data['cleaned'], i_data['I'], cv=kfolds,scoring=scoring, n_jobs=-1)

print("Accuracy:",np.mean(results_nb['test_acc']))

print("F1 Score:",np.mean(results_nb['test_f1_micro']))

print("Log Loss:",np.mean(-1*results_nb['test_neg_log_loss']))



print("\nN Component Model Performance")

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

results_nb = cross_validate(n_model, n_data['cleaned'], n_data['N'], cv=kfolds,scoring=scoring, n_jobs=-1)

print("Accuracy:",np.mean(results_nb['test_acc']))

print("F1 Score:",np.mean(results_nb['test_f1_micro']))

print("Log Loss:",np.mean(-1*results_nb['test_neg_log_loss']))



print("\nT Component Model Performance")

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

results_nb = cross_validate(t_model, t_data['cleaned'], t_data['T'], cv=kfolds,scoring=scoring, n_jobs=-1)

print("Accuracy:",np.mean(results_nb['test_acc']))

print("F1 Score:",np.mean(results_nb['test_f1_micro']))

print("Log Loss:",np.mean(-1*results_nb['test_neg_log_loss']))



print("\nJ Component Model Performance")

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

results_nb = cross_validate(j_model, j_data['cleaned'], j_data['J'], cv=kfolds,scoring=scoring, n_jobs=-1)

print("Accuracy:",np.mean(results_nb['test_acc']))

print("F1 Score:",np.mean(results_nb['test_f1_micro']))

print("Log Loss:",np.mean(-1*results_nb['test_neg_log_loss']))
probab[0][0][0]
import time

print("> > Predict your personality based on an excerpt!")

print("Submit a rather long (preferably personal) piece of writing written by you.")

print("For example, an aggregation of all of your tweets, or a personal essay.")

print("Enter 'ex' for an example. Otherwise, paste your writing here.")

print("/ / / / / / / / / / /  ")

print("Writing Sample Entry:")

time.sleep(1)

entry = pd.DataFrame()

while True:

    input_thing = input(" ")

    if input_thing == 'ex':

        print("// Example: ")

        print(data['posts'][13])

        print("")

    else:

        break

        

entry['input'] = [input_thing]

entry['cleaned'] = entry['input'].apply(cleanText)

i = int(i_model.predict(entry['cleaned']))

n = int(n_model.predict(entry['cleaned']))

t = int(t_model.predict(entry['cleaned']))

j = int(j_model.predict(entry['cleaned']))



probab = []



probab.append(i_model.predict_proba(entry['cleaned']))

probab.append(n_model.predict_proba(entry['cleaned']))

probab.append(t_model.predict_proba(entry['cleaned']))

probab.append(j_model.predict_proba(entry['cleaned']))



complete = []

comp_init = []

if i == 1:

    complete.append('Introversion')

    comp_init.append('I')

else:

    complete.append('Extroversion')

    comp_init.append('E')

if n == 1:

    complete.append('Intuition')

    comp_init.append('N')

else:

    complete.append('Sensing')

    comp_init.append('S')

if t == 1:

    complete.append('Thinking')

    comp_init.append('T')

else:

    complete.append('Feeling')

    comp_init.append('F')

if j == 1:

    complete.append('Judging')

    comp_init.append('J')

else:

    complete.append('Percieving')

    comp_init.append('P')

    

person_string = ''

for i in comp_init:

    person_string += i



print("\n/ / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / \n")

print("MBTI Personality: {}".format(person_string))



probab_simp = []

for i in range(4):

    if probab[i][0][0]>probab[i][0][1]:

        append_value = probab[i][0][0]

    else:

        append_value = probab[i][0][1]

    probab_simp.append(append_value)



'''

results = pd.DataFrame({'Characteristic':complete,'Probability':probab_simp})



print("Your MBTI characteristics are: {}".format(person_string_comp))

results'''

    

results = pd.DataFrame({'Characteristic':complete,'Abbreviation':comp_init})

results
results = pd.DataFrame({'Characteristic':complete,'Abbreviation':comp_init})

results