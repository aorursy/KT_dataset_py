import pandas as pd

import numpy as np

#from bs4 import BeautifulSoup

import requests

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords,wordnet

from nltk.tokenize import word_tokenize

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import re

import nltk

import matplotlib.pyplot as plt

import seaborn as sns

import string

from nltk import TreebankWordTokenizer, SnowballStemmer, pos_tag

import os

from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
plt.figure(figsize=(17, 5))

plt.title("Count of Personality Types")

plt.xlabel(" ")

plt.ylabel(" ")

sns.barplot(train['type'].value_counts().index, train['type'].value_counts(),palette = 'winter')

plt.show()
from sklearn.utils import resample

s = ['ENTJ', 'ENFJ','ESFP',

       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESTJ', 'ESFJ']

train_major = train[train['type'] == 'ENFP']

for a in s:

    train_min = train[train['type'] == a ]

    train_sam = resample(train_min ,replace = True, n_samples =  int(train['type'].value_counts().mean()),random_state = 123)

    train_major = pd.concat([train_major,train_sam])

    

t = ['INTJ','ENTP','INTP','INFP','INFJ'] 

for b in t :

    train_major_ds  = train[train['type'] == b ]

    train_sam = resample(train_major_ds ,replace = False, n_samples =  int(train['type'].value_counts().mean()),random_state = 123)

    train_major = pd.concat([train_major,train_sam])

    

# Upsampling



plt.figure(figsize=(17, 5))

plt.title("Resampling the Personality")

plt.xlabel(" ")

plt.ylabel(" ")

sns.barplot(train_major['type'].value_counts().index, train_major['type'].value_counts(),palette = 'winter')

plt.show()
# Mind class

train['Mind']   = train['type'].apply(lambda s : s[0])

train['Mind']   = train['Mind'].map({'I': 0,'E':1})



#Energy class

train['Energy'] = train['type'].apply(lambda s : s[1])

train['Energy'] = train['Energy'].map({'S': 0,'N':1})



#Nature

train['Nature'] = train['type'].apply(lambda s : s[2])                      

train['Nature'] = train['Nature'].map({'F': 0,'T':1})



#Tactic class

train['Tactic'] = train['type'].apply(lambda s : s[3])

train['Tactic'] = train['Tactic'].map({'P': 0,'J':1})
#Split the train data

train['posts'] = train['posts'].apply(lambda s: ' '.join(s.split('|||')))

#split the test data

test['posts'] = test['posts'].apply(lambda s: ' '.join(s.split('|||')))
train['number_of_links'] = train['posts'].str.count('http|https')
train['social_media_presence'] = train['posts'].str.count('facebook.com|face.bo|twitter.com|instagram.com|tumblr.com')
train['number_of_videos'] = train['posts'].str.count('youtube|vimeo|videobash')
train['number_of_blogs'] = train['posts'].str.count('blog|wordpress')
d = ('.png.|.jpg.|.gif|.tiff|.psd|.raw|.indd|.pdf|tinypic|.imageshack|')

train['number_of_images'] = train['posts'].str.count(d)
train.head()
plt.figure(figsize = (15,10))

sns.swarmplot('type','number_of_images', data = train,palette = 'winter')
plt.figure(figsize = (15,10))

sns.swarmplot('type','social_media_presence', data = train,palette = 'winter')
plt.figure(figsize = (15,10))

sns.swarmplot('type','number_of_videos', data = train,palette = 'winter')
plt.figure(figsize = (15,10))

sns.swarmplot('type','number_of_blogs', data = train,palette = 'winter')
plt.figure(figsize = (15,12))

sns.swarmplot('type','number_of_links', data = train,palette = 'winter')
train['posts'] = train['posts'].apply(lambda b : ''.join([ 'Photos' if s.endswith(d) else s for s in b]))

test['posts'] = test['posts'].apply(lambda b : ''.join([ 'photos' if s.endswith(d) else s for s in b]))
# Replace Web links

train['posts'] = train['posts'].apply(lambda s : ''.join([re.sub(r'http\S+', r'link', s)]))

test['posts'] = test['posts'].apply(lambda s : ''.join([re.sub(r'http\S+', r'link', s)]))
train['posts'] = train['posts'].apply(lambda s: ' '.join(s.split('::')))

train['posts'] = train['posts'].apply(lambda s: ' '.join(s.split('-')))



test['posts'] = test['posts'].apply(lambda s: ' '.join(s.split('::')))

test['posts'] = test['posts'].apply(lambda s: ' '.join(s.split('-')))
def preprocessing(text):

    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())

    text2 = " ".join("".join([" " if ch in string.digits else ch for ch in text2]).split())

    tokens = [word for sent in nltk.sent_tokenize(text2) for word in nltk.word_tokenize(sent)]

    tokens = [word.lower() for word in tokens]

    stopwds = stopwords.words('english')

    tokens = [token for token in tokens if token not in stopwds]

    tokens = [word for word in tokens if len(word)>=3]

    stemmer = SnowballStemmer('english')

    tokens = [stemmer.stem(word) for word in tokens]

    tagged_corpus = pos_tag(tokens)

    Noun_tags = ['NN','NNP','NNPS','NNS']

    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']

    lemmatizer = WordNetLemmatizer()

    def prat_lemmatize(token,tag):

        if tag in Noun_tags:

            return lemmatizer.lemmatize(token,'n')

        elif tag in Verb_tags:

            return lemmatizer.lemmatize(token,'v')

        else:

            return lemmatizer.lemmatize(token,'n')

    pre_proc_text = " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])

    return pre_proc_text
train.posts = train.posts.apply(preprocessing)

test.posts = test.posts.apply(preprocessing)
intro = ['INFJ', 'INTP', 'INTJ', 'INFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ']



for i in range(1, 9):

    for a in intro:

        plt.figure(figsize=(12, 8))

        

        df_intro = train[train.type == a]

        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(df_intro))

        plt.title(a)

        plt.imshow(wordcloud, interpolation='bilinear', aspect='equal')

        plt.axis("off")

        plt.show()

    break
entro = ['ENFJ', 'ENTP', 'ENTJ', 'ENFP', 'ESFP', 'ESTP', 'ESFJ', 'ESTJ']



for i in range(1, 9):

    for a in entro:

        plt.figure(figsize=(12, 8))

        

        df_entro = train[train.type == a]

        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(df_entro))

        plt.title(a)

        plt.imshow(wordcloud, interpolation='bilinear', aspect='equal')

        plt.axis("off")

        plt.show()

    break
train.head()
# tfidvect_sub = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.7)#, min_df=0.05)

# X_train_tfidvect_sub = tfidvect_sub.fit_transform(train['stem'])

# X_test_tfidvect_sub = tfidvect_sub.transform(test['stem'])
train.head(n=1)
# Importing TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

# Instatiating TfidfVectorizer

tfidfvectorizer = TfidfVectorizer(max_df= 0.7,stop_words='english')

# Fit transform

X_tfi = tfidfvectorizer.fit_transform(train['posts'])

# Setting the four classes

y_mind = train['Mind']

y_enegry = train['Energy']

y_nature = train['Nature']

y_tactic = train['Tactic']
# Importing CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

# Instatiating CountVectorizer

countvectorizer = CountVectorizer(max_df= 0.7,stop_words='english')

# Fit transform

X_countvectorizer = countvectorizer.fit_transform(train['posts'])
logistic_reg = LogisticRegression()

nbayes = MultinomialNB()

print('logistic Regression Parameters: ', logistic_reg.get_params())

print()

print('MultinomialNB: ', nbayes.get_params())
# Instantiate the GridSearchCV for Logistic Regression

logr_parameters = {'penalty' : ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10],'class_weight': [None, 'balanced']}

gscv_log = GridSearchCV(logistic_reg,logr_parameters)

# Instantiate the GridSearchCV for MultinomialNB

nbayes_parameters = {'alpha' : [0.1, 1, 0.1]}

gscv_naive = GridSearchCV(nbayes,nbayes_parameters)
X_train,X_test,y_train,y_test = train_test_split(X_tfi,y_enegry, test_size = 0.30,random_state = 0)
# LOGISTIC REGRESSION

logistic_reg.fit(X_train,y_train) 

# Evaluating Performance

print( 'Train score Logistic Regression ',logistic_reg.score(X_train,y_train))

y_p = logistic_reg.predict(X_test)

print('Logr Classification Report: ', classification_report(y_test,y_p))



# MULTINOMIALND

nbayes.fit(X_train,y_train)

# Evaluating Performance

print('Train score Random Forest ',nbayes.score(X_train,y_train))

y_p = nbayes.predict(X_test)

print('RF Classification Report: ', classification_report(y_test,y_p))
# LOGISTIC REGRESSION

gscv_log.fit(X_train,y_train) 

print('Logr Train score :',gscv_log.score(X_train,y_train))

y_p = gscv_log.predict(X_test)

print('Logr Accuracy :',accuracy_score(y_test,y_p))

print('Logr Best Parameters: ', gscv_log.best_params_)



# MUILTINOMILNB

gscv_naive.fit(X_train,y_train) 

print('nbayes Train score :',gscv_naive.score(X_train,y_train))

y_p = gscv_naive.predict(X_test)

print('nbayes Accuracy :',accuracy_score(y_test,y_p))

print('nbayes Best Parameters: ', gscv_naive.best_params_)
X_train,X_test,y_train,y_test = train_test_split(X_tfi,y_enegry, test_size = 0.30,random_state = 0)

# fitting the regressor

logistic_reg_tuned_E = LogisticRegression(penalty='l2', class_weight = 'balanced', C=1)

logistic_reg_tuned_E.fit(X_train,y_train) 

# Evaluating Performance

print( 'Train score Logistic Regression ',logistic_reg_tuned_E.score(X_train,y_train))

y_p = logistic_reg_tuned_E.predict(X_test)

print('Logr Classification Report: ', classification_report(y_test,y_p))



# fitting the random forest

nbayes_tuned = MultinomialNB(alpha=1)

nbayes_tuned.fit(X_train,y_train)

# Evaluating Performance

print( 'Train score Random Forest ',nbayes_tuned.score(X_train,y_train))

y_p = nbayes.predict(X_test)

print('RF Classification Report: ', classification_report(y_test,y_p))
X_train,X_test,y_train,y_test = train_test_split(X_tfi,y_mind, test_size = 0.30,random_state = 0)
# LOGISTIC REGRESSION

logistic_reg.fit(X_train,y_train) 

# Evaluating Performance

print( 'Train score Logistic Regression ',logistic_reg.score(X_train,y_train))

y_p = logistic_reg.predict(X_test)

print('Logr Classification Report: ', classification_report(y_test,y_p))



# MULTINOMIALND

nbayes.fit(X_train,y_train)

# Evaluating Performance

print('Train score Random Forest ',nbayes.score(X_train,y_train))

y_p = nbayes.predict(X_test)

print('RF Classification Report: ', classification_report(y_test,y_p))
# LOGISTIC REGRESSION

gscv_log.fit(X_train,y_train) 

print('Logr Train score :',gscv_log.score(X_train,y_train))

y_p = gscv_log.predict(X_test)

print('Logr Accuracy :',accuracy_score(y_test,y_p))

print('Logr Best Parameters: ', gscv_log.best_params_)



# MUILTINOMILNB

gscv_naive.fit(X_train,y_train) 

print('nbayes Train score :',gscv_naive.score(X_train,y_train))

y_p = gscv_naive.predict(X_test)

print('nbayes Accuracy :',accuracy_score(y_test,y_p))

print('nbayes Best Parameters: ', gscv_naive.best_params_)
X_train,X_test,y_train,y_test = train_test_split(X_tfi,y_mind, test_size = 0.30,random_state = 0)

# fitting the regressor

logistic_reg_tuned_M = LogisticRegression(penalty='l2',class_weight ='balanced', C=10)

logistic_reg_tuned_M.fit(X_train,y_train) 

# Evaluating Performance

print( 'Train score Logistic Regression ',logistic_reg_tuned_M.score(X_train,y_train))

y_p = logistic_reg_tuned_M.predict(X_test)

print('Logr Classification Report: ', classification_report(y_test,y_p))



# fitting the random forest

nbayes_tuned = MultinomialNB(alpha=1)

nbayes_tuned.fit(X_train,y_train)

# Evaluating Performance

print( 'Train score Random Forest ',nbayes_tuned.score(X_train,y_train))

y_p = nbayes.predict(X_test)

print('RF Classification Report: ', classification_report(y_test,y_p))
X_train,X_test,y_train,y_test = train_test_split(X_tfi,y_nature, test_size = 0.30,random_state = 0)
# LOGISTIC REGRESSION

logistic_reg.fit(X_train,y_train) 

# Evaluating Performance

print( 'Train score Logistic Regression ',logistic_reg.score(X_train,y_train))

y_p = logistic_reg.predict(X_test)

print('Logr Classification Report: ', classification_report(y_test,y_p))



# MULTINOMIALND

nbayes.fit(X_train,y_train)

# Evaluating Performance

print('Train score Random Forest ',nbayes.score(X_train,y_train))

y_p = nbayes.predict(X_test)

print('RF Classification Report: ', classification_report(y_test,y_p))
# LOGISTIC REGRESSION

gscv_log.fit(X_train,y_train) 

print('Logr Train score :',gscv_log.score(X_train,y_train))

y_p = gscv_log.predict(X_test)

print('Logr Accuracy :',accuracy_score(y_test,y_p))

print('Logr Best Parameters: ', gscv_log.best_params_)



# MUILTINOMILNB

gscv_naive.fit(X_train,y_train) 

print('nbayes Train score :',gscv_naive.score(X_train,y_train))

y_p = gscv_naive.predict(X_test)

print('nbayes Accuracy :',accuracy_score(y_test,y_p))

print('nbayes Best Parameters: ', gscv_naive.best_params_)
X_train,X_test,y_train,y_test = train_test_split(X_tfi,y_nature, test_size = 0.30,random_state = 0)

# fitting the regressor

logistic_reg_tuned_N = LogisticRegression(penalty='l2',class_weight = 'balanced' , C=10)

logistic_reg_tuned_N.fit(X_train,y_train) 

# Evaluating Performance

print( 'Train score Logistic Regression ',logistic_reg_tuned_N.score(X_train,y_train))

y_p = logistic_reg_tuned_N.predict(X_test)

print('Logr Classification Report: ', classification_report(y_test,y_p))



# fitting the random forest

nbayes_tuned = MultinomialNB(alpha=1)

nbayes_tuned.fit(X_train,y_train)

# Evaluating Performance

print( 'Train score Random Forest ',nbayes_tuned.score(X_train,y_train))

y_p = nbayes.predict(X_test)

print('RF Classification Report: ', classification_report(y_test,y_p))
X_train,X_test,y_train,y_test = train_test_split(X_tfi,y_tactic, test_size = 0.30,random_state = 0)
# LOGISTIC REGRESSION

logistic_reg.fit(X_train,y_train) 

# Evaluating Performance

print( 'Train score Logistic Regression ',logistic_reg.score(X_train,y_train))

y_p = logistic_reg.predict(X_test)

print('Logr Classification Report: ', classification_report(y_test,y_p))



# MULTINOMIALND

nbayes.fit(X_train,y_train)

# Evaluating Performance

print('Train score Random Forest ',nbayes.score(X_train,y_train))

y_p = nbayes.predict(X_test)

print('RF Classification Report: ', classification_report(y_test,y_p))
# LOGISTIC REGRESSION

gscv_log.fit(X_train,y_train) 

print('Logr Train score :',gscv_log.score(X_train,y_train))

y_p = gscv_log.predict(X_test)

print('Logr Accuracy :',accuracy_score(y_test,y_p))

print('Logr Best Parameters: ', gscv_log.best_params_)



# MUILTINOMILNB

gscv_naive.fit(X_train,y_train) 

print('nbayes Train score :',gscv_naive.score(X_train,y_train))

y_p = gscv_naive.predict(X_test)

print('nbayes Accuracy :',accuracy_score(y_test,y_p))

print('nbayes Best Parameters: ', gscv_naive.best_params_)
X_train,X_test,y_train,y_test = train_test_split(X_tfi,y_tactic, test_size = 0.30,random_state = 0)

# fitting the regressor

logistic_reg_tuned_T = LogisticRegression(penalty='l2', class_weight = 'balanced', C=1)

logistic_reg_tuned_T.fit(X_train,y_train) 

# Evaluating Performance

print( 'Train score Logistic Regression ',logistic_reg_tuned_T.score(X_train,y_train))

y_p = logistic_reg_tuned_T.predict(X_test)

print('Logr Classification Report: ', classification_report(y_test,y_p))



# fitting the random forest

nbayes_tuned = MultinomialNB(alpha=1)

nbayes_tuned.fit(X_train,y_train)

# Evaluating Performance

print( 'Train score Random Forest ',nbayes_tuned.score(X_train,y_train))

y_p = nbayes.predict(X_test)

print('RF Classification Report: ', classification_report(y_test,y_p))
logistic_reg = LogisticRegression(class_weight = 'balanced', )
# Main Prediction

X_test_m = tfidfvectorizer.transform(test['posts'])

# Mind

logistic_reg.fit(X_tfi,y_mind)

y_mind_pred = logistic_reg.predict(X_test_m)
# Main Prediction



# Energy

logistic_reg.fit(X_tfi,y_enegry)

y_energy_pred = logistic_reg.predict(X_test_m)
# Main Prediction



# Tactics

logistic_reg.fit(X_tfi,y_tactic)

y_tactic_pred = logistic_reg.predict(X_test_m)
#Main Prediction

# Nature

logistic_reg.fit(X_tfi,y_nature)

y_nature_pred = logistic_reg.predict(X_test_m)
df = pd.DataFrame({'Id':test['id'],'mind':y_mind_pred,'energy':y_energy_pred,'nature':y_nature_pred,'tactics':y_tactic_pred})



df.to_csv('out_csv.csv',index = False)