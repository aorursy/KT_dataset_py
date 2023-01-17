#Importing pandas and numpy

import pandas as pd

import numpy as np
#Importing the training and test data

train = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/train.csv')

test = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/test.csv')
#Training data info

train.info()
#first 5 rows

train.head()
#Test data info

test.info()
#first 5 rows of test data

test.head()
#Frequency plot of classes

import matplotlib.pyplot as plt

import seaborn as sns

print(sns.countplot(train['label'],label=True))

plt.title('Class Distribution')
#Visualizing the classes of train data

chat_data = train['label'].value_counts()

plt.pie(chat_data, autopct='%1.1f%%', shadow=True,labels=['Negative Class','Positive Class'])

plt.title('Class Distribution');

plt.show()
#Creating the length column for tweet

train['pre_clean_len']=  [len(t) for t in train.tweet]
#Box plot of all data

fig, ax = plt.subplots(figsize=(5, 5))

plt.boxplot(train.pre_clean_len)

plt.title('Word length of all tweets ')

plt.show()
#Box plot of positive data

fig, ax = plt.subplots(figsize=(5, 5))

plt.boxplot(train[train['label']==0].pre_clean_len)

plt.title('Word Length of Positive Tweets')

plt.show()
#Box plot of negative data

fig, ax = plt.subplots(figsize=(5, 5))

plt.boxplot(train[train['label']==1].pre_clean_len)

plt.title('Word Length of Negative Tweets')

plt.show()
#Let's look at exact numbers of positive and negative tweet length

print('\033[5m'+'Positive Tweets:'+"\033[0;0m")

print('Minimum number of words are',train[train['label']==1].pre_clean_len.min())

print('Maximum number of words are',train[train['label']==1].pre_clean_len.max())

print(' ')

print('\033[5m'+'Negative Tweets:'+"\033[0;0m")

print('Minimum number of words are',train[train['label']==0].pre_clean_len.min())

print('Maximum number of words are',train[train['label']==0].pre_clean_len.max())
#Word cloud of all tweets

from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(str(train['tweet']))

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title('Word Cloud - All tweets',fontsize=20,fontweight='bold')

plt.show()
#Word cloud of negative tweets

from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(str(train[train['label']==0]['tweet']))

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title('Word Cloud - Positive tweets',fontsize=20,fontweight='bold')

plt.show()
#Word cloud of positive tweets

from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(str(train[train['label']==1]['tweet']))

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title('Word Cloud - Negative tweets',fontsize=20,fontweight='bold')

plt.show()
#Negative tweets

print(train[train['label']==1]['tweet'][13])

print(train[train['label']==1]['tweet'][77])

print(train[train['label']==1]['tweet'][111])

print(train[train['label']==1]['tweet'][263])
#Positive tweets

print(train[train['label']==0]['tweet'][1])

print(train[train['label']==0]['tweet'][33])

print(train[train['label']==0]['tweet'][31943])

print(train[train['label']==0]['tweet'][21])
# ------------Step 1 - Definig cleaning functions - URLs, Mentions, Negation handling, UF8 (BOM), Special chracters and numbers

#!pip install bs4

#!pip install nltk

#!pip install et_xmlfile



#!pip install lxml

import re

from bs4 import BeautifulSoup

from nltk.tokenize import WordPunctTokenizer

Tokenz = WordPunctTokenizer()

Mentions_Removal = r'@[A-Za-z0-9_]+'

Http_Removal = r'http(s?)://[^ ]+'

#HttpS_Removal = r'https://[^ ]+'

Www_Removal = r'www.[^ ]+'



#Combining the above 3 removals functions

#Combining_MentnHttp = r'|'.join((Mentions_Removal,Http_Removal))

Combining_MentnHttp1 = r'|'.join((Http_Removal,Www_Removal))





#Creating a negation dictionary because words with apostrophe symbol (') will (Can't > can t) 

Negation_Dictonary = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", 

                "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", 

                "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", 

                "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", 

                "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",

                "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 

                "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", 

                "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",

                "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 

                "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 

                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 

                "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", 

                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",

                "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",

                "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",

                "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",

                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 

                "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",

                "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  

                "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 

                "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", 

                "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", 

                "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 

                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",

                 "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

Negation_Joining= re.compile(r'\b(' + '|'.join(Negation_Dictonary.keys()) + r')\b')



def clean_tweet_function(text):

    BeautifulSoup_assign = BeautifulSoup(text, 'html.parser')

    Souping = BeautifulSoup_assign.get_text()

    try:

        BOM_removal = Souping.decode("utf-8-sig").replace(u"\ufffd", "?")

    except:

        BOM_removal = Souping

    Comb_2 = re.sub(Combining_MentnHttp1, '', BOM_removal)

    #Comb_3 = re.sub(Www_Removal,'',Comb_2)

    Comb_3 = re.sub(Mentions_Removal,'',Comb_2)

    LowerCase = Comb_3.lower()

    Negation_Handling = Negation_Joining.sub(lambda x: Negation_Dictonary[x.group()], LowerCase)

    Letters_only = re.sub("[^a-zA-Z]", " ", Negation_Handling)

    

    # Removing unneccessary white- Tokenizing and joining together

    Tokenization = [x for x  in Tokenz.tokenize(Letters_only) if len(x) > 1]

    return (" ".join(Tokenization)).strip()

clean_tweet_function



#Cleaning up the data with step 1

xrange = range #Defining X range
%%time

xrange = range

print ("Cleaning tweets in train data...\n")

clean_tweet_train = []

for i in xrange(0,len(train)):

    if( (i+1)%100000 == 0 ):

        "Reviews %d of %d has been processed".format( i+1, len(train) )  

        

    clean_tweet_train.append(clean_tweet_function(train['tweet'][i]))

    

#Changing into dataframe

train['cleaned_tweet'] = clean_tweet_train
%%time

xrange = range

print ("Cleaning tweets in test data...\n")

clean_tweet_test = []

for i in xrange(0,len(test)):

    if( (i+1)%100000 == 0 ):

        "Reviews %d of %d has been processed".format( i+1, len(test) )  

        

    clean_tweet_test.append(clean_tweet_function(test['tweet'][i]))

    

#Changing into dataframe

test['cleaned_tweet'] = clean_tweet_test
#Lets compare the positive tweets before and after cleaning

print('BEFORE - ',train[train['label']==1]['tweet'][13])

print('AFTER - ',train[train['label']==1]['cleaned_tweet'][13])

print('')



print('BEFORE - ',train[train['label']==1]['tweet'][77])

print('AFTER - ',train[train['label']==1]['cleaned_tweet'][77])

print('')

print('BEFORE - ',train[train['label']==1]['tweet'][111])

print('AFTER - ',train[train['label']==1]['cleaned_tweet'][111])
#Lets compare the positive tweets before and after cleaning

print('BEFORE - ',train[train['label']==0]['tweet'][1])

print('AFTER - ',train[train['label']==0]['cleaned_tweet'][1])

print('')



print('BEFORE - ',train[train['label']==0]['tweet'][33])

print('AFTER - ',train[train['label']==0]['cleaned_tweet'][33])

print('')

print('BEFORE - ',train[train['label']==0]['tweet'][31943])

print('AFTER - ',train[train['label']==0]['cleaned_tweet'][31943])
#Importing stop words and removing negative words from it

from nltk.corpus import stopwords

stopwords = set(stopwords.words('english')) - {'no', 'nor', 'not'} #we don't Stopwords to remove negation from our tweets



def remove_stopwords(text):

    return ' '.join([word for word in str(text).split() if word not in stopwords])



#Removing stop words from training and test

train['cleaned_tweet'] = train['cleaned_tweet'].apply(lambda text: remove_stopwords(text))

test['cleaned_tweet'] = test['cleaned_tweet'].apply(lambda text: remove_stopwords(text))
#Defining x and y

X = train['cleaned_tweet']

y = train['label']



X_test = test['cleaned_tweet']
#TFIDF bi-gram 

#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

'''

tfidf = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english').

'''
# TFIDF tri-gram

'''

tfidf = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')

'''
#Importing TFIDF 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf = TfidfVectorizer(min_df=3,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 4), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')
#Fitting TFIDF to both training and test

x_train_tfidf =  tfidf.fit_transform(X) 

x_test_tfidf = tfidf.transform(X_test)
import warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

import time

start_time = time.time()

param_grid = {'C': np.arange(20,30,2),

              'max_iter': np.arange(100,1200,100),

              'penalty': ['l1','l2']}



i=1

kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)

for train_index,test_index in kf.split(x_train_tfidf,y):

    print('\n{} of kfold {}'.format(i,kf.n_splits))

    xtr,xvl = x_train_tfidf[train_index],x_train_tfidf[test_index]

    ytr,yvl = y[train_index],y[test_index]

    

    model = RandomizedSearchCV(estimator=LogisticRegression(class_weight='balanced'),param_distributions=param_grid,verbose=0)

    



    model.fit(xtr, ytr)

    #print (model.best_params_)

    pred=model.predict(xvl)

    print('roc_auc_score',roc_auc_score(yvl,pred))

    i+=1



print("Execution time: " + str((time.time() - start_time)) + ' ms')

print ('best parameters',model.best_params_)
roc_auc_logistic = roc_auc_score(yvl,pred).mean()

f1_logistic = f1_score(yvl,pred).mean()

print('Mean - ROC AUC', roc_auc_logistic)

print('F1 Score - ', f1_logistic)

print('Confusion Matrix \n',confusion_matrix(yvl,pred))
#DecisionTree with tuned hyperparameters

from sklearn.tree import DecisionTreeClassifier

start_time = time.time()

param_grid = {'criterion': ['gini','entropy'],

             'min_samples_split':[50,70,100,150],

             'max_features': ['sqrt','log2']}





i=1

kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)

for train_index,test_index in kf.split(x_train_tfidf,y):

    print('\n{} of kfold {}'.format(i,kf.n_splits))

    xtr,xvl = x_train_tfidf[train_index],x_train_tfidf[test_index]

    ytr,yvl = y[train_index],y[test_index]

    

    model = RandomizedSearchCV(estimator=DecisionTreeClassifier(class_weight={0:1,1:5}),param_distributions=param_grid,verbose=0)

    



    model.fit(xtr, ytr)

    #print (model.best_params_)

    pred=model.predict(xvl)

    print('roc_auc_score',roc_auc_score(yvl,pred))

    i+=1



print("Execution time: " + str((time.time() - start_time)) + ' ms')

print ('best parameters',model.best_params_)
#Model Accuracy

roc_auc_dt = roc_auc_score(yvl,pred).mean()

f1_dt = f1_score(yvl,pred).mean()

print('Mean - ROC AUC', roc_auc_dt)

print('F1 Score - ', f1_dt)

print('Confusion Matrix \n',confusion_matrix(yvl,pred))
from sklearn.ensemble import RandomForestClassifier

start_time = time.time()

param_grid = {'criterion': ['entropy'],

             'min_samples_split':np.arange(10,100,20),

             'max_features': ['sqrt'],

             'n_estimators':[10,20,30]}



i=1

kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)

for train_index,test_index in kf.split(x_train_tfidf,y):

    print('\n{} of kfold {}'.format(i,kf.n_splits))

    xtr,xvl = x_train_tfidf[train_index],x_train_tfidf[test_index]

    ytr,yvl = y[train_index],y[test_index]

    

    model = RandomizedSearchCV(estimator=RandomForestClassifier(),param_distributions=param_grid,verbose=0)

    



    model.fit(xtr, ytr)

    #print (model.best_params_)

    pred=model.predict(xvl)

    print('roc_auc_score',roc_auc_score(yvl,pred))

    i+=1



print("Execution time: " + str((time.time() - start_time)) + ' ms')

print ('best parameters',model.best_params_)
#Model Accuracy

roc_auc_rf = roc_auc_score(yvl,pred).mean()

f1_rf = f1_score(yvl,pred).mean()

print('Mean - ROC AUC', roc_auc_rf)

print('F1 Score - ', f1_rf)

print('Confusion Matrix \n',confusion_matrix(yvl,pred))
from xgboost import XGBClassifier

start_time = time.time()

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5],

        'learning_rate': [0.01,0.1,0.7,1],

        'eval_metric': ['auc']

        }





i=1

kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)

for train_index,test_index in kf.split(x_train_tfidf,y):

    print('\n{} of kfold {}'.format(i,kf.n_splits))

    xtr,xvl = x_train_tfidf[train_index],x_train_tfidf[test_index]

    ytr,yvl = y[train_index],y[test_index]

    

    model = RandomizedSearchCV(estimator=XGBClassifier(min_scale_weight=12,n_estimators=600),param_distributions=params,verbose=0)

    



    model.fit(xtr, ytr)

    #print (model.best_params_)

    pred=model.predict(xvl)

    print('roc_auc_score',roc_auc_score(yvl,pred))

    i+=1



print("Execution time: " + str((time.time() - start_time)) + ' ms')

print ('best parameters',model.best_params_)
#Model Accuracy

roc_auc_xg = roc_auc_score(yvl,pred).mean()

f1_xg = f1_score(yvl,pred).mean()

print('Mean - ROC AUC', roc_auc_xg)

print('F1 Score - ', f1_xg)

print('Confusion Matrix \n',confusion_matrix(yvl,pred))
from sklearn.ensemble import AdaBoostClassifier

start_time = time.time()

#params = {'n_estimators':[100,300,600]}

i=1

kf = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)

for train_index,test_index in kf.split(x_train_tfidf,y):

    print('\n{} of kfold {}'.format(i,kf.n_splits))

    xtr,xvl = x_train_tfidf[train_index],x_train_tfidf[test_index]

    ytr,yvl = y[train_index],y[test_index]

    

    model = AdaBoostClassifier()



    model.fit(xtr, ytr)

    #print (model.best_params_)

    pred=model.predict(xvl)

    print('roc_auc_score',roc_auc_score(yvl,pred))

    print('Confusion Matrix \n',confusion_matrix(yvl,pred))

    i+=1



print("Execution time: " + str((time.time() - start_time)) + ' ms')
#Model Accuracy

roc_auc_ada = roc_auc_score(yvl,pred).mean()

f1_ada = f1_score(yvl,pred).mean()

print('Mean - ROC AUC', roc_auc_ada)

print('F1 Score - ', f1_ada)

print('Confusion Matrix \n',confusion_matrix(yvl,pred))
import lightgbm as lgb

start_time = time.time()

params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'auc',

    'num_leaves': 31,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0

    }

i=1

kf = StratifiedKFold(n_splits=10,random_state=42,shuffle=True)

for train_index,test_index in kf.split(x_train_tfidf,y):

    print('\n{} of kfold {}'.format(i,kf.n_splits))

    xtr,xvl = x_train_tfidf[train_index],x_train_tfidf[test_index]

    ytr,yvl = y[train_index],y[test_index]

    

    train_set = lgb.Dataset(xtr, label=ytr)

    val_set = lgb.Dataset(xvl, label=yvl)

    

    model = lgb.train(params,train_set, valid_sets=val_set, verbose_eval=500)



    #print (model.best_params_)

    pred=model.predict(xvl)

    print('roc_auc_score',roc_auc_score(yvl,pred))

    #print('Confusion Matrix \n',confusion_matrix(yvl,pred))

    i+=1



print("Execution time: " + str((time.time() - start_time)) + ' ms')

print('')
#pred_noemb_val_y = model.predict([xvl], batch_size=1024, verbose=1)

from sklearn import metrics

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(yvl, (pred>thresh).astype(int))))
#Model accuracy

roc_auc_lgb = roc_auc_score(yvl,pred).mean()

print('Mean - ROC AUC', roc_auc_lgb)

pred1 = np.where(pred > 0.29, 1, 0)

f1_lgb =  f1_score(yvl,pred1).mean()

print('F1 Score - ',f1_lgb)
#Summary table for all models



results = pd.DataFrame({

    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest','XG Boost', 'Ada Boosting','LGB'],

    'Mean - ROC AUC Score (Fold=10)': [roc_auc_logistic, roc_auc_dt, roc_auc_rf,roc_auc_xg,roc_auc_ada,roc_auc_lgb],

    'Mean - F1 Score': [f1_logistic,f1_dt,f1_rf,f1_xg,f1_ada,f1_lgb]})

results