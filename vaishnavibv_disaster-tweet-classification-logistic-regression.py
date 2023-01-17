#Imports:

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import math

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from matplotlib import rcParams

from wordcloud import WordCloud
pd.set_option('display.max_columns', None) 

pd.set_option('display.max_rows', None)  

pd.set_option('display.max_colwidth', -1) 

#Code:

# reading the csv file into pandas dataframes

df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df.head()
df['target'].value_counts()
#creating a new column- length 

# this gives the length of the post

df['length'] = np.NaN

for i in range(0,len(df['text'])):

    df['length'][i]=(len(df['text'][i]))

df.length = df.length.astype(int)
#creating subplots to see distribution of length of tweet

sns.set_style("darkgrid");

f, (ax1, ax2) = plt.subplots(figsize=(12,6),nrows=1, ncols=2,tight_layout=True);

sns.distplot(df[df['target']==1]["length"],bins=30,ax=ax1);

sns.distplot(df[df['target']==0]["length"],bins=30,ax=ax2);

ax1.set_title('\n Distribution of length of tweet labelled Disaster\n');

ax2.set_title('\nDistribution of length of tweet labelled No Disaster\n ');

ax1.set_ylabel('Frequency');
# word cloud for words related to Disaster 

text=" ".join(post for post in df[df['target']==1].text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words related to Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()
# word cloud for words related to No Disaster 

text=" ".join(post for post in df[df['target']==0].text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words related to No Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()
#calculating basline accuracy

df['target'].value_counts(normalize=True)
# Import Tokenizer

from nltk.tokenize import RegexpTokenizer
# Instantiate Tokenizer

tokenizer = RegexpTokenizer(r'\w+') 

#changing the contents of selftext to lowercase

df.loc[:,'text'] = df.text.apply(lambda x : str.lower(x))
#removing hyper link, latin characters and digits

df['text']=df['text'].str.replace('http.*.*', '',regex = True)

df['text']=df['text'].str.replace('û.*.*', '',regex = True)

df['text']=df['text'].str.replace(r'\d+','',regex= True)
# "Run" Tokenizer

df['tokens'] = df['text'].map(tokenizer.tokenize)
#displaying first 5 rows of dataframe

df.head()
# Printing English stopwords

print(stopwords.words("english"))
#assigning stopwords to a variable

stop = stopwords.words("english")
# adding this stop word to list of stopwords as it appears on frequently occuring word

item=['amp'] #'https','co','http','û','ûò','ûó','û_'
stop.extend(item)
#removing stopwords from tokens

df['tokens']=df['tokens'].apply(lambda x: [item for item in x if item not in stop])
# Importing lemmatizer 

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



# Instantiating lemmatizer 

lemmatizer = WordNetLemmatizer()

lemmatize_words=[]

for i in range (len(df['tokens'])):

    word=''

    for j in range(len(df['tokens'][i])):

        lemm_word=lemmatizer.lemmatize(df['tokens'][i][j])#lemmatize

        

        word=word + ' '+lemm_word # joining tokens into sentence    

    lemmatize_words.append(word) # store in list

   
#creating a new column to store the result

df['lemmatized']=lemmatize_words
#displaying first 5 rows of dataframe

df.head()
#imports

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
#defining X and y for the model

X = df['lemmatized']

y = df['target']
# Spliting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
#ensuring that the value counts are quite evenly distributed

y_train.value_counts()
y_test.shape
# pipeline will consist of two stages:

# 1.Instantiating countVectorizer

# 2.Instantiating logistic regression model



pipe = Pipeline([

    ('cvec', CountVectorizer()),  

    ('lr', LogisticRegression()) 

])
tuned_params = {

    'cvec__max_features': [2500, 3000, 3500],

    'cvec__min_df': [2,3],

    'cvec__max_df': [.9, .95],

    'cvec__ngram_range': [(1,1), (1,2)]

}

gs = GridSearchCV(pipe, param_grid=tuned_params, cv=3) # Evaluating model on unseen data



model_lr=gs.fit(X_train, y_train) # Fitting model



# This is the average of all cv folds for a single 

#combination of the parameters specified in the tuned_params

print(gs.best_score_) 



#displaying the best values of parameters

gs.best_params_
# Test score

gs.score(X_train, y_train)
# Test score

gs.score(X_test, y_test)
# Generating predictions!

predictions_lr = model_lr.predict(X_test)
# Importing the confusion matrix function

from sklearn.metrics import confusion_matrix
# Generating confusion matrix

confusion_matrix(y_test, predictions_lr)
#interpreting confusion matrix

tn, fp, fn, tp = confusion_matrix(y_test, predictions_lr).ravel()
#values with coreesponding labels

print("True Negatives: %s" % tn)

print("False Positives: %s" % fp)

print("False Negatives: %s" % fn)

print("True Positives: %s" % tp)
# Importing model

from sklearn.naive_bayes import MultinomialNB
# Instantiating model

nb = MultinomialNB()
# Instantiating CountVectorizer.

cvec = CountVectorizer(max_features = 500)
# fit_transform() fits the model and transforms training data into feature vectors

X_train_cvec = cvec.fit_transform(X_train, y_train).todense()
#tranform test data and convert into array

X_test_cvec = cvec.transform(X_test).todense()
# Fitting model

model_nb=nb.fit(X_train_cvec, y_train)
# Generating predictions

predictions_nb = model_nb.predict(X_test_cvec)
# Training score

model_nb.score(X_train_cvec, y_train)
# Test score

model_nb.score(X_test_cvec, y_test)
# Generating confusion matrix

confusion_matrix(y_test, predictions_nb)
#interpreting confusion matrix

tn, fp, fn, tp = confusion_matrix(y_test, predictions_nb).ravel()
#values with coreesponding labels

print("True Negatives: %s" % tn)

print("False Positives: %s" % fp)

print("False Negatives: %s" % fn)

print("True Positives: %s" % tp)
# word cloud for Frequntly occuring words related to Disaster

text=" ".join(post for post in df[df['target']==1].lemmatized)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words related to Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()
# word cloud for Frequntly occuring words related to No Disaster

text=" ".join(post for post in df[df['target']==0].lemmatized)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words related to No Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()
#reading the test data

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test.head()
#creating a new column- length 

# this gives the length of the post

test['length'] = np.NaN

for i in range(0,len(test['text'])):

    test['length'][i]=(len(test['text'][i]))

test.length = test.length.astype(int)
# word cloud for Frequntly occuring words in test dataframe

text=" ".join(post for post in df.text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words in test dataframe \n\n',fontsize=18)

plt.axis("off")

plt.show()
# Instantiate Tokenizer

tokenizer = RegexpTokenizer(r'\w+')
#changing the contents of selftext to lowercase

test.loc[:,'text'] = test.text.apply(lambda x : str.lower(x))
#removing hyper link and latin characters

test['text']=test['text'].str.replace('http.*.*', '',regex = True)

test['text']=test['text'].str.replace('û.*.*', '',regex = True)

test['text']=test['text'].str.replace(r'\d+','',regex= True)
# "Run" Tokenizer

test['tokens'] = test['text'].map(tokenizer.tokenize)
#displaying first 5 rows of dataframe

test.head()
#removing stopwords from tokens

test['tokens']=test['tokens'].apply(lambda x: [item for item in x if item not in stop])
lemmatize_words=[]

for i in range (len(test['tokens'])):

    word=''

    for j in range(len(test['tokens'][i])):

        lemm_word=lemmatizer.lemmatize(test['tokens'][i][j])#lemmatize

        

        word=word + ' '+lemm_word # joining tokens into sentence    

    lemmatize_words.append(word) # store in list

   
#creating a new column to store the result

test['lemmatized']=lemmatize_words
#displaying first 5 rows of dataframe

test.head()
# word cloud for Frequntly occuring words in test dataframe after lemmatizing

text=" ".join(post for post in df.lemmatized)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words in test dataframe \n\n',fontsize=18)

plt.axis("off")

plt.show()
predictions_kaggle = model_lr.predict(test['lemmatized'])
#tranform test data and convert into array

kaggle_cvec = cvec.transform(test['lemmatized']).todense()
predictions_kaggle_nb=model_nb.predict(kaggle_cvec)
# Creating an empty data frame

submission_kaggle = pd.DataFrame()
# Assigning values to the data frame-submission_kaggle

submission_kaggle['Id'] = test.id

submission_kaggle['target'] = predictions_kaggle
# Head of submission_kaggle

submission_kaggle.head()
# saving data as  final_kaggle.csv

submission_kaggle.loc[ :].to_csv('final_kaggle.csv',index=False)
# Creating an empty data frame

submission_kaggle_nb = pd.DataFrame()
# Assigning values to the data frame-submission_kaggle

submission_kaggle_nb['Id'] = test.id

submission_kaggle_nb['target'] = predictions_kaggle_nb
# Head of submission_kaggle

submission_kaggle_nb.head()
# saving data as  final_kaggle.csv

submission_kaggle_nb.loc[ :].to_csv('final_kaggle_nb.csv',index=False)