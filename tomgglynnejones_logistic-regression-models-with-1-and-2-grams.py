import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

print(train.head())

print('\n', train.info())
train['author'] = train.author.astype('category')

train['author'] = train.author.cat.codes

print(train.head())



plt.hist(train['author'], bins = [0, 0.5, 1, 1.5, 2, 2.5], color='blue')

plt.xlabel('Author')

plt.ylabel('Count')

plt.xticks([0.25, 1.25, 2.25], ['EAP', 'HPL', 'MWS'])

plt.show()
public = pd.read_csv('../input/test.csv')

print(public.head())
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    train.text, train.author, test_size=0.3,

    random_state=42, stratify=train.author)



d_train = {'text': X_train.values, 'author': y_train.values}

train_df = pd.DataFrame(data=d_train)



d_test = {'text': X_test.values, 'author': y_test.values}

test_df = pd.DataFrame(data=d_test)
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from string import punctuation
sentence_0 = train_df['text'][0]

tokenized_0 = word_tokenize(sentence_0)

print('Original:\n', sentence_0)

print('\n Tokenized (', len(tokenized_0), 'words ): \n', tokenized_0)
stopwords_list = stopwords.words('english') + list(punctuation)

print(stopwords_list)
stripped_0 = [word.lower() for word in tokenized_0 if word.lower() not in stopwords_list]

print('Stopwords removed (', len(stripped_0), 'words ): \n', stripped_0)
from nltk.stem import PorterStemmer



stemmer = PorterStemmer()



stemmed_0 = [stemmer.stem(word) for word in stripped_0]

print('Stemmed (', len(stemmed_0), 'words ): \n', stemmed_0)
def text_to_stemmed(input_text):

    '''Dependencies: ntlk.tokenize.word_tokenize, ntlk.corpus.stopwords,

    ntlk.stem.PorterStemmer, string.punctuation must be imported'''

    tokenized = word_tokenize(input_text)

    stopwords_list = stopwords.words('english') + list(punctuation)

    stripped = [word.lower() for word in tokenized if word.lower() not in stopwords_list]

    stemmed = [PorterStemmer().stem(word) for word in stripped]

    space = ' '

    stemmed_str = space.join(stemmed)

    return stemmed_str

    
train_df['stemmed'] = train_df['text'].apply(text_to_stemmed)

print(train_df['stemmed'][0])
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



count_vec = CountVectorizer()

train_vec = count_vec.fit_transform(train_df['stemmed'])



tfidf = TfidfTransformer()

train_tfidf = tfidf.fit_transform(train_vec)
test_df['stemmed'] = test_df['text'].apply(text_to_stemmed)



test_vec = count_vec.transform(test_df['stemmed'])

test_tfidf = tfidf.transform(test_vec)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import log_loss, accuracy_score



nb = MultinomialNB()

nb.fit(train_tfidf, train_df['author'].values)



y_pred = nb.predict(test_tfidf)

y_prob = nb.predict_proba(test_tfidf)

print('Prediction accuracy for naive Bayes:', accuracy_score(test_df['author'], y_pred))

print('Log loss for naive Bayes:', log_loss(test_df['author'], y_prob))
def export_predictions(classifier, name, test_set):

    '''test_set should be a set of processed features from the public data

    for input to the classifier. 

    name should be a .csv filename (string)'''

    temp_probs = classifier.predict_proba(test_set)

    

    temp_df = public

    

    temp_df['EAP'] = temp_probs[:, 0]

    temp_df['HPL'] = temp_probs[:, 1]

    temp_df['MWS'] = temp_probs[:, 2]



    temp_df.to_csv(name, columns=['id', 'EAP', 'HPL', 'MWS'], index=False)
public['stemmed'] = public['text'].apply(text_to_stemmed)

public_vec = count_vec.transform(public['stemmed'])

public_tfidf = tfidf.transform(public_vec)



export_predictions(nb, 'naive_bayes_stemmed.csv', public_tfidf)
from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



ll = make_scorer(log_loss, greater_is_better=False, needs_proba=True)



gs_params = {'C':[0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]}

lr = LogisticRegression()

lr_cv = GridSearchCV(lr, gs_params, scoring=ll)



lr_cv.fit(train_tfidf, train_df['author'].values)



y_pred_lr = lr_cv.predict(test_tfidf)

y_prob_lr = lr_cv.predict_proba(test_tfidf)



print('prediction accuracy for Logistic Regression (GridSearchCV):',

      accuracy_score(test_df['author'], y_pred_lr))

print('log loss for Logistic Regression (GridSearchCV):',

      log_loss(test_df['author'], y_prob_lr))

print('Best regularization parameter: ',

      lr_cv.cv_results_['params'][lr_cv.best_index_])
count_vec_b = CountVectorizer(ngram_range=(1, 2))

tfidf_b = TfidfTransformer()



train_vec_b = count_vec_b.fit_transform(train_df['stemmed'])

train_tfidf_b = tfidf_b.fit_transform(train_vec_b)



test_vec_b = count_vec_b.transform(test_df['stemmed'])

test_tfidf_b = tfidf_b.transform(test_vec_b)
gs_params = {'C':[0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]}

lr_b = LogisticRegression()

lr_cv_b = GridSearchCV(lr, gs_params, scoring=ll)



lr_cv_b.fit(train_tfidf_b, train_df['author'].values)



y_pred_lr_b = lr_cv_b.predict(test_tfidf_b)

y_prob_lr_b = lr_cv_b.predict_proba(test_tfidf_b)



print('prediction accuracy for Logistic Regression with bigrams (GridSearchCV):',

      accuracy_score(test_df['author'], y_pred_lr_b))

print('log loss for Logistic Regression with bigrams (GridSearchCV):',

      log_loss(test_df['author'], y_prob_lr_b))

print('Best regularization parameter: ',

      lr_cv_b.cv_results_['params'][lr_cv_b.best_index_])
count_vect_b = CountVectorizer(ngram_range=(1, 2))

tfidf_b_2 = TfidfTransformer()



train_vec_b_2 = count_vect_b.fit_transform(train_df['text'])

train_tfidf_b_2 = tfidf_b_2.fit_transform(train_vec_b_2)



test_vec_b_2 = count_vect_b.transform(test_df['text'])

test_tfidf_b_2 = tfidf_b_2.transform(test_vec_b_2)



public_vec_b_2 = count_vect_b.transform(public['text'])

public_tfidf_b_2 = tfidf_b_2.transform(public_vec_b_2)



print(train_tfidf_b.shape)

print(train_tfidf_b_2.shape)
gs_params = {'C':[0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]}



lr = LogisticRegression()



lr_cv_b_2 = GridSearchCV(lr, gs_params, scoring=ll)

lr_cv_b_2.fit(train_tfidf_b_2, train_df['author'].values)



y_pred_lr_b_2 = lr_cv_b_2.predict(test_tfidf_b_2)

y_prob_lr_b_2 = lr_cv_b_2.predict_proba(test_tfidf_b_2)



print('prediction accuracy for Logistic Regression with bigrams (GridSearchCV):',

      accuracy_score(test_df['author'], y_pred_lr_b_2))

print('log loss for Logistic Regression with bigrams (GridSearchCV):',

      log_loss(test_df['author'], y_prob_lr_b_2))

print('Best regularization parameter: ',

      lr_cv_b_2.cv_results_['params'][lr_cv_b_2.best_index_])
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.pipeline import Pipeline



kbest = SelectKBest(chi2)

lr = LogisticRegression()

pipe = Pipeline([('kbest', kbest), ('lr', lr)])



gs_params = {

    'kbest__k': [60000, 80000, 100000, 120000, 140000],

    'lr__C':[30.0, 100.0, 300, 1000]

}



kbest_lr_cv = GridSearchCV(pipe, gs_params, scoring=ll)



kbest_lr_cv.fit(train_tfidf_b_2, train_df['author'].values)



y_pred_kbest_lr_cv = kbest_lr_cv.predict(test_tfidf_b_2)

y_prob_kbest_lr_cv = kbest_lr_cv.predict_proba(test_tfidf_b_2)



print('prediction accuracy for Logistic Regression with bigrams, kbest (GridSearchCV):',

      accuracy_score(test_df['author'], y_pred_kbest_lr_cv))

print('log loss for Logistic Regression with bigrams, kbest (GridSearchCV):',

      log_loss(test_df['author'], y_prob_kbest_lr_cv))

print('Best regularization parameter: ',

      kbest_lr_cv.cv_results_['params'][kbest_lr_cv.best_index_])