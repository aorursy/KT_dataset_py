#Installing the pyspellchecker library

!pip install pyspellchecker



#Loading the libraries used in this notebook

import numpy as np

import pandas as pd



from string import punctuation

import re

import nltk

from spellchecker import SpellChecker

from sklearn.feature_extraction.text import CountVectorizer



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')



from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import GridSearchCV



#Loading the data

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train[train.target == 1].head(10)
train[train.target == 0].head(10)
print('The number of rows in the training set is {}.\nMissing values in each column:\n'.format(train.shape[0]))

print(train.isna().sum())
print('There are {} unique values in the keyword column, which are given by\n'.format(len(train['keyword'].unique().tolist())))

print(train['keyword'].unique().tolist())
print('Percentage of disaster tweets with the \'oil%20spill\' keyword: {}'\

      .format(100*train[train.keyword=='oil%20spill']['target'].sum()/train[train.keyword=='oil%20spill'].shape[0]))

train[train.keyword=='oil%20spill'].head()
print('Percentage of disaster tweets with the \'thunder\' keyword: {}'\

      .format(100*train[train.keyword=='thunder']['target'].sum()/train[train.keyword=='thunder'].shape[0]))

train[train.keyword=='thunder'].head()
print('There are {} unique values in the location column. The first 10 of them are given by\n'.format(len(train['location'].unique().tolist())))

print(train['location'].unique().tolist()[1:11])
print('Percentage of disasters if location is null: {}'.format(\

    100*train[train['location'].isna()]['target'].sum()/train[train['location'].isna()]['target'].shape[0]))

print('Percentage of disasters if location is not null: {}'.format(\

    100*train[~train['location'].isna()]['target'].sum()/train[~train['location'].isna()]['target'].shape[0]))
_ = sns.countplot(train['target'])
X_train = train.copy()

X_train['keyword'] = X_train['keyword'].fillna('none') #Filling null values



#CountVectorizer discarding english common stopwords and using a vocabulary with at most 1000 words

vect = CountVectorizer(max_features=1000,stop_words='english')

vect.fit(X_train.text)

text_features = vect.transform(X_train.text)

X_train[vect.get_feature_names()] = pd.DataFrame(text_features.toarray(),columns=vect.get_feature_names())



#Target variable

y_train = X_train['target']



#One-hot-encoding and dropping columns

X_train = pd.get_dummies(X_train,columns=['keyword']).drop(columns=['id','location','text','target'])
#Ridge Classifier

clf = RidgeClassifier()



alpha = [0.03,0.1,0.3,1,3,10,30]

param_grid = dict(alpha=alpha)



grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3,scoring='f1_macro')



grid_search_result = grid_search.fit(X_train,y_train)



best_score, best_params = grid_search_result.best_score_,grid_search_result.best_params_

print("Ridge Classifier F1 score: %f using %s" % (best_score, best_params))
#Naive Bayes classifier

clf = MultinomialNB()



alpha = [1,0.8,0.6,0.4,0.2]

param_grid = dict(alpha=alpha)



grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3,scoring='f1_macro')



grid_search_result = grid_search.fit(X_train,y_train)



best_score, best_params = grid_search_result.best_score_,grid_search_result.best_params_

print("Naive Bayes F1 score: %f using %s" % (best_score, best_params))
X_test = test.copy()

X_test['keyword'] = X_test['keyword'].fillna('none')



text_features = vect.transform(X_test.text)



X_test[vect.get_feature_names()] = pd.DataFrame(text_features.toarray(),columns=vect.get_feature_names())



X_test = pd.get_dummies(X_test,columns=['keyword']).drop(columns=['id','location','text'])

X_test = X_test[X_train.columns.tolist()]



y_pred = grid_search_result.predict(X_test)

output = pd.DataFrame({'id': test.id, 'target': y_pred})

output.to_csv('simple_NB_submission.csv', index=False)

print("Your submission was successfully saved!")

pd.read_csv('simple_NB_submission.csv').head()
fig,ax = plt.subplots(1,2,figsize=(16,5))

train[train.target == 1]['text'].str.len().hist(ax=ax[0],bins=20)

train[train.target == 0]['text'].str.len().hist(color='blue',ax=ax[1],bins=20)

ax[0].set_title('Disaster')

ax[0].set_xlabel('Tweet size')

ax[0].set_ylabel('Counts')

ax[1].set_title('Not a disaster')

ax[1].set_xlabel('Tweet size')

ax[1].set_ylabel('Counts')



plt.show()
def most_common_words(tweets,n_words = 10,stop_words = None):

    vect = CountVectorizer(max_features=n_words,stop_words = stop_words)

    vect.fit(tweets)

    X = vect.transform(tweets)

    X_df = pd.DataFrame(X.toarray(),columns=vect.get_feature_names())

    most_common_words = X_df.sum().sort_values()

    return most_common_words
fig,ax = plt.subplots(1,2,figsize=(16,5))

most_common_words(train.text).plot(kind='barh',ax=ax[0])

most_common_words(train.text,stop_words='english').plot(kind='barh',ax=ax[1])

ax[0].set_title('With stopwords')

ax[0].set_xlabel('Counts')

ax[0].set_ylabel('Word')

ax[1].set_title('Without stopwords (english)')

ax[1].set_xlabel('Counts')

ax[1].set_ylabel('Word')



plt.show()
fig,ax = plt.subplots(1,2,figsize=(16,5))

most_common_words(train[train.target==1].text,stop_words='english').plot(kind='barh',ax=ax[0])

most_common_words(train[train.target==0].text,stop_words='english').plot(kind='barh',ax=ax[1],color='blue')

ax[0].set_title('Disaster')

ax[0].set_xlabel('Counts')

ax[0].set_ylabel('Word')

ax[1].set_title('Not a disaster')

ax[1].set_xlabel('Counts')

ax[1].set_ylabel('Word')



plt.show()
X_train = train.copy()

X_train['keyword'] = X_train['keyword'].fillna('none')



vect = CountVectorizer(max_features=1000,stop_words='english',token_pattern=r"\w\w+|!|\?|#|@")

vect.fit(X_train.text)

text_features = vect.transform(X_train.text)



X_train[vect.get_feature_names()] = pd.DataFrame(text_features.toarray(),columns=vect.get_feature_names())



y_train = X_train['target']

X_train = pd.get_dummies(X_train,columns=['keyword']).drop(columns=['id','location','text','target'])



clf = MultinomialNB()



alpha = [1,0.8,0.6,0.4,0.2]

param_grid = dict(alpha=alpha)



grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3,scoring='f1_macro')



grid_search_result = grid_search.fit(X_train,y_train)



best_score, best_params = grid_search_result.best_score_,grid_search_result.best_params_

print("Naive Bayes F1 score: %f using %s" % (best_score, best_params))
emoji_pattern = re.compile("["

                        u"\U0001F600-\U0001F64F"  # emoticons

                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                        u"\U0001F680-\U0001F6FF"  # transport & map symbols

                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                        u"\U00002702-\U000027B0"

                        u"\U000024C2-\U0001F251"

                        "]+", flags=re.UNICODE)



train['emojis'] = train['text'].apply(lambda x: len(re.findall(emoji_pattern,x)))

print('total number of emojis on tweets: ',train['emojis'].sum())
train = train.drop(columns=['emojis'])



hashtag_count = {}

for row in train['text']:

    hashtags = re.findall('#\w+',row)

    for hashtag in hashtags:

        if hashtag.lower() not in hashtag_count:

            hashtag_count[hashtag.lower()] = 1

        else:

            hashtag_count[hashtag.lower()] += 1



hashtag_count_series = pd.Series(hashtag_count)

_ = hashtag_count_series.sort_values(ascending=False)[:10].sort_values().plot(kind='barh')

_ = plt.xlabel('Counts')
hashtag_count_disaster = {}

hashtag_count_not_disaster = {}



for row in train[train.target==1]['text']:

    hashtags = re.findall('#\w+',row)

    for hashtag in hashtags:

        if hashtag.lower() not in hashtag_count_disaster:

            hashtag_count_disaster[hashtag.lower()] = 1

        else:

            hashtag_count_disaster[hashtag.lower()] += 1

            

            

for row in train[train.target==0]['text']:

    hashtags = re.findall('#\w+',row)

    for hashtag in hashtags:

        if hashtag.lower() not in hashtag_count_not_disaster:

            hashtag_count_not_disaster[hashtag.lower()] = 1

        else:

            hashtag_count_not_disaster[hashtag.lower()] += 1

            

hashtag_count_disaster_series = pd.Series(hashtag_count_disaster)

hashtag_count_not_disaster_series = pd.Series(hashtag_count_not_disaster)



fig,ax = plt.subplots(1,2,figsize=(16,5))

hashtag_count_disaster_series.sort_values(ascending=False)[:10].sort_values().plot(kind='barh',ax=ax[0])

hashtag_count_not_disaster_series.sort_values(ascending=False)[:10].sort_values().plot(kind='barh',ax=ax[1],color='blue')

ax[0].set_title('Disaster')

ax[0].set_xlabel('Counts')

ax[1].set_title('Not a disaster')

ax[1].set_xlabel('Counts')

plt.show()
mention_count = {}

for row in train['text']:

    mentions = re.findall('@\w+',row)

    for mention in mentions:

        if mention.lower() not in mention_count:

            mention_count[mention.lower()] = 1

        else:

            mention_count[mention.lower()] += 1

            

mention_count_series = pd.Series(mention_count)

_ = mention_count_series.sort_values(ascending=False)[:10].sort_values().plot(kind='barh')

_ = plt.xlabel('Counts')
mention_count_disaster = {}

mention_count_not_disaster = {}



for row in train[train.target==1]['text']:

    mentions = re.findall('@\w+',row)

    for mention in mentions:

        if mention.lower() not in mention_count_disaster:

            mention_count_disaster[mention.lower()] = 1

        else:

            mention_count_disaster[mention.lower()] += 1

            

            

for row in train[train.target==0]['text']:

    mentions = re.findall('@\w+',row)

    for mention in mentions:

        if mention.lower() not in mention_count_not_disaster:

            mention_count_not_disaster[mention.lower()] = 1

        else:

            mention_count_not_disaster[mention.lower()] += 1

            

mention_count_disaster_series = pd.Series(mention_count_disaster)

mention_count_not_disaster_series = pd.Series(mention_count_not_disaster)



fig,ax = plt.subplots(1,2,figsize=(16,5))

mention_count_disaster_series.sort_values(ascending=False)[:10].sort_values().plot(kind='barh',ax=ax[0])

mention_count_not_disaster_series.sort_values(ascending=False)[:10].sort_values().plot(kind='barh',ax=ax[1],color='blue')

ax[0].set_title('Disaster')

ax[0].set_xlabel('Counts')

ax[1].set_title('Not a disaster')

ax[1].set_xlabel('Counts')

plt.show()
#Counting URLs

train['URLs'] = train['text'].apply(lambda x: len(re.findall(r'http\S+|www\.\S+',x)))



#Removing URLs

url = re.compile(r'http\S+|www\.\S+')

train['text'] = train['text'].apply(lambda x: url.sub(r'',x))



#Extracting tweet sizes

train['text_size'] = train['text'].str.len()



#Counting punctuations, hashtags, and mentions

train['!'] = train['text'].apply(lambda x: len(re.findall('!',x)))

train['?'] = train['text'].apply(lambda x: len(re.findall('\?',x)))

train['#'] = train['text'].apply(lambda x: len(re.findall('#',x)))

train['@'] = train['text'].apply(lambda x: len(re.findall('@',x)))



#Counting unique hashtags that appear in at least 0.1% of the tweets

for hashtag in hashtag_count_series[hashtag_count_series >= 7].index:

    train[hashtag] = train['text'].apply(lambda x: len(re.findall(hashtag,x.lower())))

    

#Counting unique mentions that appear in at least 0.1% of the tweets    

for mention in mention_count_series[mention_count_series >= 7].index:

    train[mention] = train['text'].apply(lambda x: len(re.findall(mention,x.lower())))



#Removing hashtags and mentions

hashtags_and_mentions = re.compile(r'@\w+|#\w+')

train['text'] = train['text'].apply(lambda x: hashtags_and_mentions.sub(r'',x))



#Defining a function that remove stopwords

def remove_stopwords(text,stop_words):

    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text



#Counting words that do not belong to the stopwords from the nltk corpus (english)

stop_words = nltk.corpus.stopwords.words("english")

train['text_number_words'] = train['text'].apply(lambda x: len(re.findall(r'\w+',remove_stopwords(x.lower(),stop_words))))



#Counting capital words

train['capital_words'] = train['text'].apply(lambda x: len(re.findall(r'[A-Z][A-Z]+',x)))



#Counting unknown words using pyspellchecker

spell = SpellChecker()

train['unknown_words'] = train['text'].apply(lambda x: len(spell.unknown(re.findall(r'\w+',x.lower()))))
#Defining the stemmer

stemmer = nltk.stem.LancasterStemmer()



#Function that splits and stems a keyword from the keyword column

def keyword_stemming(keyword,stemmer):

    stemmed_keyword = ''

    for word in keyword.split('%20'):

        stemmed_keyword = stemmed_keyword + ' ' + stemmer.stem(word)

    return stemmed_keyword[1:]



train['keyword'] = train['keyword'].fillna('none').astype(str)

train['keyword'] = train['keyword'].apply(lambda x: keyword_stemming(x,stemmer)).astype('category')



print('there are {} unique values in the keyword column, which are given by\n'.format(len(train['keyword'].unique().tolist())))

print(train['keyword'].unique().tolist())
fig,ax = plt.subplots(1,2,figsize=(16,5))

sns.countplot(x='!',data=train,hue='target',hue_order=[1,0],ax=ax[0])

sns.countplot(x='?',data=train,hue='target',hue_order=[1,0],ax=ax[1])

ax[0].set_xlabel('Number of !')

ax[0].set_ylabel('Counts')

ax[1].set_xlabel('Number of ?')

ax[1].set_ylabel('Counts')



plt.show()
fig,ax = plt.subplots(1,2,figsize=(16,5))

sns.countplot(x='#',data=train,hue='target',hue_order=[1,0],ax=ax[0])

sns.countplot(x='@',data=train,hue='target',hue_order=[1,0],ax=ax[1])

ax[0].set_xlabel('Number of #')

ax[0].set_ylabel('Counts')

ax[1].set_xlabel('Number of @')

ax[1].set_ylabel('Counts')



plt.show()
fig,ax = plt.subplots(1,2,figsize=(16,5))

sns.countplot(x='capital_words',data=train,hue='target',hue_order=[1,0],ax=ax[0])

sns.countplot(x='unknown_words',data=train,hue='target',hue_order=[1,0],ax=ax[1])

ax[0].set_xlabel('Number of capital words')

ax[0].set_ylabel('Counts')

ax[1].set_xlabel('Number of unknown words')

ax[1].set_ylabel('Counts')



plt.show()
X_train = train.copy()



vect_text = CountVectorizer(max_features=1000,stop_words='english')

vect_text.fit(X_train.text)

text_features = vect_text.transform(X_train.text)

X_train[vect_text.get_feature_names()] = pd.DataFrame(text_features.toarray(),columns=vect_text.get_feature_names())



vect_keyword = CountVectorizer(stop_words='english')

vect_keyword.fit(X_train.keyword)

keyword_features = vect_keyword.transform(X_train.keyword)

X_train[vect_keyword.get_feature_names()] = pd.DataFrame(keyword_features.toarray(),columns=vect_keyword.get_feature_names())



y_train = X_train['target']

X_train = X_train.drop(columns=['id','keyword','location','text','target'])



clf = MultinomialNB()



alpha = [1,0.8,0.6,0.4,0.2]

param_grid = dict(alpha=alpha)



grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3,scoring='f1_macro')



grid_search_result = grid_search.fit(X_train,y_train)



best_score, best_params = grid_search_result.best_score_,grid_search_result.best_params_

print("Naive Bayes F1 score: %f using %s" % (best_score, best_params))
test['URLs'] = test['text'].apply(lambda x: len(re.findall(r'http\S+|www\.\S+',x)))

test['text'] = test['text'].apply(lambda x: url.sub(r'',x))

test['text_size'] = test['text'].str.len()

test['!'] = test['text'].apply(lambda x: len(re.findall('!',x)))

test['?'] = test['text'].apply(lambda x: len(re.findall('\?',x)))

test['#'] = test['text'].apply(lambda x: len(re.findall('#',x)))

test['@'] = test['text'].apply(lambda x: len(re.findall('@',x)))



for hashtag in hashtag_count_series[hashtag_count_series >= 7].index:

    test[hashtag] = test['text'].apply(lambda x: len(re.findall(hashtag,x.lower())))

    

for mention in mention_count_series[mention_count_series >= 7].index:

    test[mention] = test['text'].apply(lambda x: len(re.findall(mention,x.lower())))



test['text'] = test['text'].apply(lambda x: hashtags_and_mentions.sub(r'',x))

test['text_number_words'] = test['text'].apply(lambda x: len(re.findall(r'\w+',remove_stopwords(x.lower(),stop_words))))

test['capital_words'] = test['text'].apply(lambda x: len(re.findall(r'[A-Z][A-Z]+',x)))

test['unknown_words'] = test['text'].apply(lambda x: len(spell.unknown(re.findall(r'\w+',x.lower()))))

test['keyword'] = test['keyword'].fillna('none').astype(str)

test['keyword'] = test['keyword'].apply(lambda x: keyword_stemming(x,stemmer)).astype('category')
X_test = test.copy()

text_features = vect_text.transform(X_test.text)

X_test[vect_text.get_feature_names()] = pd.DataFrame(text_features.toarray(),columns=vect_text.get_feature_names())

keyword_features = vect_keyword.transform(X_test.keyword)

X_test[vect_keyword.get_feature_names()] = pd.DataFrame(keyword_features.toarray(),columns=vect_keyword.get_feature_names())



X_test = X_test.drop(columns=['id','keyword','location','text'])

X_test = X_test[X_train.columns.tolist()]



y_pred = grid_search_result.predict(X_test)

output = pd.DataFrame({'id': test.id, 'target': y_pred})

output.to_csv('metadata_NB_submission.csv', index=False)

print("Your submission was successfully saved!")

pd.read_csv('metadata_NB_submission.csv').head()
def train_NB_clf(df,max_df=0.5,min_df=2,stopwords=None,max_ngram=1,alpha=[1]):

    X_train = df.copy()



    vect_text = CountVectorizer(max_df = max_df, min_df = min_df,stop_words = stopwords,ngram_range = (1,max_ngram))

    vect_text.fit(df.text)

    text_features = vect_text.transform(X_train.text)



    vect_keyword = CountVectorizer(stop_words='english')

    vect_keyword.fit(X_train.keyword)

    keyword_features = vect_keyword.transform(X_train.keyword)



    y_train = X_train['target']

    X_train = X_train.drop(columns=['id','keyword','location','text','target'])



    X_train[vect_text.get_feature_names()] = pd.DataFrame(text_features.toarray(),columns=vect_text.get_feature_names())

    X_train[vect_keyword.get_feature_names()] = pd.DataFrame(keyword_features.toarray(),columns=vect_keyword.get_feature_names())



    clf = MultinomialNB()

    param_grid = dict(alpha=alpha)

    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=3,scoring='f1_macro')

    grid_search_result = grid_search.fit(X_train,y_train)

    

    return vect_text,vect_keyword,grid_search_result
vect_text,vect_keyword,grid_search_result = train_NB_clf(train,stopwords='english')

print("Naive Bayes F1 score: %f" % (grid_search_result.best_score_))
vect_text,vect_keyword,grid_search_result = train_NB_clf(train,stopwords='english',max_ngram=2)

print("Naive Bayes F1 score using 1-grams and 2-grams: %f" % (grid_search_result.best_score_))

vect_text,vect_keyword,grid_search_result = train_NB_clf(train,stopwords='english',max_ngram=3)

print("Naive Bayes F1 score using 1-grams, 2-grams, and 3-grams: %f" % (grid_search_result.best_score_))
vect_text,vect_keyword,grid_search_result = train_NB_clf(train,max_ngram=1)

print("Naive Bayes F1 score using only 1-grams: %f" % (grid_search_result.best_score_))

vect_text,vect_keyword,grid_search_result = train_NB_clf(train,max_ngram=2)

print("Naive Bayes F1 score using 1-grams and 2-grams: %f" % (grid_search_result.best_score_))

vect_text,vect_keyword,grid_search_result = train_NB_clf(train,max_ngram=3)

print("Naive Bayes F1 score using 1-grams, 2-grams, and 3-grams: %f" % (grid_search_result.best_score_))
vect_text,vect_keyword,grid_search_result = train_NB_clf(train,max_ngram=2)



X_test = test.copy()

text_features = vect_text.transform(X_test.text)

keyword_features = vect_keyword.transform(X_test.keyword)



X_test = X_test.drop(columns=['id','keyword','location','text'])



X_test[vect_text.get_feature_names()] = pd.DataFrame(text_features.toarray(),columns=vect_text.get_feature_names())

X_test[vect_keyword.get_feature_names()] = pd.DataFrame(keyword_features.toarray(),columns=vect_keyword.get_feature_names())



y_pred = grid_search_result.predict(X_test)

output = pd.DataFrame({'id': test.id, 'target': y_pred})

output.to_csv('ngrams_NB_submission.csv', index=False)

print("Your submission was successfully saved!")

pd.read_csv('ngrams_NB_submission.csv').head()
def text_processor(text,stemmer):

    text = text.lower()

    tokenized_text = nltk.word_tokenize(text)

    tokenized_text = [word for word in tokenized_text if word not in punctuation]

    stemmed_text = ''

    for word in tokenized_text:

        #The stemmer transforms news to new, but we do not want that.

        if word != 'news':

            stemmed_text = stemmed_text + ' ' + stemmer.stem(word)

        else:

            stemmed_text = stemmed_text + ' ' + word

    return stemmed_text[1:]



#Testing the new function

text_processor('Hi, I am learning #NLP, and I\' not sure if this transformation will be good to this problem...',stemmer)
#Applying the pre-processing to the training and test datasets

train['text'] = train['text'].apply(lambda x: text_processor(x,stemmer))

test['text'] = test['text'].apply(lambda x: text_processor(x,stemmer))
vect_text,vect_keyword,grid_search_result = train_NB_clf(train,max_ngram=1)

print("Naive Bayes F1 score using only 1-grams: %f" % (grid_search_result.best_score_))

vect_text,vect_keyword,grid_search_result = train_NB_clf(train,max_ngram=2)

print("Naive Bayes F1 score using 1-grams and 2-grams: %f" % (grid_search_result.best_score_))
X_test = test.copy()

text_features = vect_text.transform(X_test.text)

keyword_features = vect_keyword.transform(X_test.keyword)



X_test = X_test.drop(columns=['id','keyword','location','text'])



X_test[vect_text.get_feature_names()] = pd.DataFrame(text_features.toarray(),columns=vect_text.get_feature_names())

X_test[vect_keyword.get_feature_names()] = pd.DataFrame(keyword_features.toarray(),columns=vect_keyword.get_feature_names())



y_pred = grid_search_result.predict(X_test)

output = pd.DataFrame({'id': test.id, 'target': y_pred})

output.to_csv('ngrams_stemming_NB_submission.csv', index=False)

print("Your submission was successfully saved!")

pd.read_csv('ngrams_stemming_NB_submission.csv').head()