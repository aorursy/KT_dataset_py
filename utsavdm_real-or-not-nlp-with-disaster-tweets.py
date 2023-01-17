# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display

import tqdm
train_dataset = pd.read_csv('../input/nlp-getting-started/train.csv')

print("Training dataset:\n")

display(train_dataset)
# checking if there are any duplicates:

dup_rows = train_dataset[train_dataset.duplicated(['keyword','location','text','target'])]

display(dup_rows)



# eliminating the duplicates:

train_dataset.drop_duplicates(['keyword','location','text','target'], keep='first', inplace=True)

display(train_dataset)

print("Duplicates removed")
# eliminating junk values from 'keyword' in train set:

empty_str=[]

for i in tqdm.tqdm(train_dataset['keyword']):

    if i is np.NaN:

        i=''

    empty_str.append(i)

train_dataset['keyword'] = empty_str



# eliminating junk values from 'location' in train set:

empty_str=[]

for i in tqdm.tqdm(train_dataset['location']):

    if i is np.NaN:

        i=''

    empty_str.append(i)

train_dataset['location'] = empty_str



display(train_dataset)
new_train_dataset = train_dataset

details=[]



for i in tqdm.tqdm(train_dataset['text']):

    details.append(str(train_dataset.text[train_dataset.text == i].values[0])+" "+

                       str(train_dataset.keyword[train_dataset.text == i].values[0])+" "+

                           str(train_dataset.location[train_dataset.text == i].values[0]))



# creating the new column 'details':

new_train_dataset['details'] = details

display(new_train_dataset)
# import statements:

import re   # to search for html tags, punctuations & special characters



# importing gensim.models to implement Word2Vec:

import gensim

from gensim import models

from gensim.models import Word2Vec

from gensim.models import KeyedVectors



final_clean_details = []

list_of_lists_details = []



# Remove HTML tags - getting all the HTML tags and replacing them with blank spaces:

def cleanhtml(sentence):

    clean_text = re.sub('<.*?>', ' ', sentence)

    return clean_text



# Remove punctuations & special characters - getting all the punctuations and replacing them with blank spaces:

def cleanpunc(sentence):

    clean_text = re.sub(r'[@#$%\^&\*+=\d]', r' ', sentence) # removing special characters

    clean_text = re.sub(r'[,.;\'"\-\!?:\\/|\[\]{}()]', r' ', clean_text) # removing punctuations

    return clean_text



final_clean_details = []



for sentence in tqdm.tqdm(new_train_dataset['details'].values):

    sentence = cleanhtml(sentence)

    sentence = cleanpunc(sentence)

    clean_sentence = []

    

    # for each word in the sentence, if it is alphabetic, we append it to the new list

    for word in sentence.split():

        if word.isalpha() and len(word)>2:

            clean_sentence.append(word.lower())

    

    cleansent = " ".join(clean_sentence)

    # for each review in the 'Text' column, we create a list of words that appear in that sentence and store it in another list. 

    # basically, a list of lists - because that's how the model takes the input while training:

    final_clean_details.append(cleansent)

    list_of_lists_details.append(clean_sentence)



# creating another column with clean text:

new_train_dataset['clean_details'] = final_clean_details

print("Sentence cleaning completed")

display(new_train_dataset)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



# Bag of Words (BoW)/ uni-gram:

bow_model = CountVectorizer(min_df=3)

bow_model = bow_model.fit(new_train_dataset['clean_details'])

bow_vectors = bow_model.transform(new_train_dataset['clean_details'])

print("Shape of the BoW vectors: ", bow_vectors.shape)



# bi-gram:

gram_model = CountVectorizer(min_df=3, ngram_range=(1,2))

gram_vectors = gram_model.fit_transform(new_train_dataset['clean_details'])

print("Shape of the bi-gram vectors: ", gram_vectors.shape)



# TF-IDF (Term frequency - Inverse Document Frequency):

tfidf_model = TfidfVectorizer(min_df=3, ngram_range=(1,3))

tfidf_model = tfidf_model.fit(new_train_dataset['clean_details'])

tfidf_vectors = tfidf_model.transform(new_train_dataset['clean_details'])

print("Shape of the TF-IDF vectors: ", tfidf_vectors.shape)
# importing gensim.models to implement Word2Vec:

import gensim

from gensim import models

from gensim.models import Word2Vec

from gensim.models import KeyedVectors



# in the filename, 300 stands for 300 dimensions:

# loading the model, this is a very high resource consuming task: 

g_trained_model = KeyedVectors.load_word2vec_format('../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin', binary=True)

print("Model loaded successfully: ", type(g_trained_model))
w2v_vec_sentence= []



for sentence in tqdm.tqdm(list_of_lists_details):

    vec_sent = np.zeros(300)

    for word in sentence:

        if word in g_trained_model:

            vec_word = g_trained_model[word]

            vec_sent+=vec_word



    vec_sent/=len(sentence)

    w2v_vec_sentence.append(vec_sent)



print("Avg. Word2Vec calculations completed!")
# creating a dictionary to store the IDF values and access them directly afterwards:

tfidf_dict = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))



tfidf_vec_sentence = []

tfidf_features = tfidf_model.get_feature_names()



for sentence in tqdm.tqdm(list_of_lists_details):

    vec_sent = np.zeros(300)

    length_sent = len(sentence)

    sum_tfidf_w2v_val = 0

    sum_tfidf_val = 0

    for word in sentence:

        if word in g_trained_model and word in tfidf_dict.keys():

            tfidf_val = (sentence.count(word)/ length_sent) * tfidf_dict[word]  # calc.  TF * IDF

            w2v_val = g_trained_model[word]  # calc.  Word2Vec

            tfidf_w2v_val = tfidf_val * w2v_val  # calc. TF-IDF * Word2Vec

            sum_tfidf_w2v_val += tfidf_w2v_val # summation of TF-IDF * Word2Vec

            sum_tfidf_val += tfidf_val  # summation of TF * IDF

         

    if sum_tfidf_val == 0:

            tfidf_vec_sentence.append(vec_sent)

    try:

        tfidf_vec_sentence.append(sum_tfidf_w2v_val/ sum_tfidf_val) # (summation of TF-IDF * Word2Vec)/ (summation of TF * IDF)

    except:

        pass

    

print("TF-IDF weighted Word2Vec calculations completed!")
# Merging 'id' & 'target' for Avg. Word2Vec for train dataset:

w2v_vec_array = np.asarray(w2v_vec_sentence)

print(type(w2v_vec_array))

print(w2v_vec_array.shape)



#w2v_vec_array = np.hstack((w2v_vec_array, new_train_dataset['id'].values.reshape(7561,1)))

#w2v_vec_array = np.hstack((w2v_vec_array, new_train_dataset['target'].values.reshape(7561,1)))

#print(w2v_vec_array.shape)
target = new_train_dataset['target']

target = target.values.reshape(7561,1)

print(type(target))

print(target.shape)
# K-NN for Average Word2Vec:



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Splitting into training set & test set:

x_train, x_test, y_train, y_test = train_test_split(w2v_vec_array, target, test_size=0.2, random_state=0)

print("Training set:")

print(x_train.shape)

print("Testing set: ")

print(x_test.shape)



# creating cross validation set out of train set:

# x_train, x_cv, y_train, y_cv = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# print("Training set:")

# print(x_train.shape)

# print("Validation set:")

# print(x_cv.shape)



# NOTE - Try to split the x_train & y_train into x_cv, x_cv_test, y_cv & y_cv_test respectively since we are 

#        trying to find the best hyper-parameter 'aplha'



for i in range(1,20,2):

    knn_classifier = KNeighborsClassifier(n_neighbors=i)

    knn_classifier.fit(x_train, y_train.ravel())

    pred_cv = knn_classifier.predict(x_test)

    acc = accuracy_score(y_test, pred_cv)

    print("Cross validation accuracy of Avg. weighted Word2Vec vectors using ", i,"NN is: ", acc)
# using 11-NN

knn_classifier = KNeighborsClassifier(n_neighbors=11)

knn_classifier = knn_classifier.fit(x_train, y_train)

knn_preds = knn_classifier.predict(x_test)



# Checking with confusion matrix:

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, knn_preds)

import seaborn as sns

print("Confusion matrix:")

sns.heatmap(data=cm, annot=True)
TP = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]

TN = cm[1][1]



print("True positives: ", TP)

print("True negatives: ", TN)

print("False positives: ", FP)

print("False negatives: ", FN)



acc = (TP+TN)/(TP+TN+FP+FN)

precision = TP/ (TP+FP)

recall = TP/ (TP+FN)

f1_score = 2*precision*recall/ (precision + recall)



print("\n=========Final performance evaluation=========")

print("Accuracy: ", acc)

print("Precision: ", precision)

print("Recall: ", recall)

print("F-1 Score: ", f1_score)

print("==============================================")
from sklearn.linear_model import LogisticRegression



lr_classifier = LogisticRegression(dual=False, random_state=0, solver='liblinear')

lr_classifier = lr_classifier.fit(x_train, y_train.ravel())

lr_preds = lr_classifier.predict(x_test)



# Checking with confusion matrix:

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, lr_preds)

import seaborn as sns

print("Confusion matrix:")

sns.heatmap(data=cm, annot=True)
TP = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]

TN = cm[1][1]



print("True positives: ", TP)

print("True negatives: ", TN)

print("False positives: ", FP)

print("False negatives: ", FN)



acc = (TP+TN)/(TP+TN+FP+FN)

precision = TP/ (TP+FP)

recall = TP/ (TP+FN)

f1_score = 2*precision*recall/ (precision + recall)



print("\n=========Final performance evaluation=========")

print("Accuracy: ", acc)

print("Precision: ", precision)

print("Recall: ", recall)

print("F-1 Score: ", f1_score)

print("==============================================")
from sklearn.svm import SVC



linSVM = SVC(kernel='rbf', random_state=0)

linSVM = linSVM.fit(x_train, y_train.ravel())

linSVM_preds = lr_classifier.predict(x_test)



# Checking with confusion matrix:

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, linSVM_preds)

import seaborn as sns

print("Confusion matrix:")

sns.heatmap(data=cm, annot=True)
TP = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]

TN = cm[1][1]



print("True positives: ", TP)

print("True negatives: ", TN)

print("False positives: ", FP)

print("False negatives: ", FN)



acc = (TP+TN)/(TP+TN+FP+FN)

precision = TP/ (TP+FP)

recall = TP/ (TP+FN)

f1_score = 2*precision*recall/ (precision + recall)



print("\n=========Final performance evaluation=========")

print("Accuracy: ", acc)

print("Precision: ", precision)

print("Recall: ", recall)

print("F-1 Score: ", f1_score)

print("==============================================")
from sklearn.naive_bayes import BernoulliNB



# Splitting into training set & test set:

x_train, x_test, y_train, y_test = train_test_split(bow_vectors, target, test_size=0.2, random_state=0)

print("Training set:")

print(x_train.shape)

print("Testing set: ")

print(x_test.shape)



# NOTE - Try to split the x_train & y_train into x_cv, x_cv_test, y_cv & y_cv_test respectively since we are 

#        trying to find the best hyper-parameter 'aplha'.



for i in range(1,10):

    nb_class = BernoulliNB(alpha=i)

    nb_class = nb_class.fit(x_train, y_train.ravel())

    nb_preds = nb_class.predict(x_test)

    acc = accuracy_score(y_test, nb_preds)

    print("Cross validation accuracy of Avg. weighted Word2Vec vectors using Bernoulli NB, with aplha=", i," is: ", acc)
nb_class = BernoulliNB(alpha=2)

nb_class = nb_class.fit(x_train, y_train)

nb_preds = nb_class.predict(x_test)



# Checking with confusion matrix:

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, nb_preds)

import seaborn as sns

print("Confusion matrix:")

sns.heatmap(data=cm, annot=True)
TP = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]

TN = cm[1][1]



print("True positives: ", TP)

print("True negatives: ", TN)

print("False positives: ", FP)

print("False negatives: ", FN)



acc = (TP+TN)/(TP+TN+FP+FN)

precision = TP/ (TP+FP)

recall = TP/ (TP+FN)

f1_score = 2*precision*recall/ (precision + recall)



print("\n=========Final performance evaluation=========")

print("Accuracy: ", acc)

print("Precision: ", precision)

print("Recall: ", recall)

print("F-1 Score: ", f1_score)

print("==============================================")