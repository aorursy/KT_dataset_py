

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from sklearn.pipeline import Pipeline

from textblob import TextBlob

from sklearn.tree import DecisionTreeClassifier 

  

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Step 1: Load data



messages = pd.read_csv('/kaggle/input/SMSSpamCollection.txt', sep='\t',names=["label", "message"])  

print (messages)



# view aggregate statistics 

messages.groupby('label').describe()
#How long are the messages?

messages['length'] = messages['message'].map(lambda text: len(text))

print (messages.head())
messages.length.plot(bins=20, kind='hist')

messages.length.describe()

print (list(messages.message[messages.length > 900]))





messages.hist(column='length', by='label', bins=50)

#Step 2: Data preprocessing



#tokenized





def ensureUtf(s):

  try:

      if type(s) == unicode:

        return s.encode('utf8', 'ignore')

  except: 

    return str(s)

def split_into_tokens(message):

#     message = unicode(message, 'utf-8')  # convert bytes into proper unicode

#    message ='\U0001F642'

     ensureUtf(message)

     return TextBlob(message).words



messages.message.head()



messages.message.head().apply(split_into_tokens)

#With textblob, we'd detect part-of-speech (POS) tags with:

TextBlob("Hello world, how is it going?").tags



# normalize words into their base form (lemmas) with

def split_into_lemmas(message):

#    message = unicode(message, 'utf8').lower()

    ensureUtf(message)

    words = TextBlob(message).words

    # for each word, take its "base form" = lemma 

    return [word.lemma for word in words]



messages.message.head().apply(split_into_lemmas)
#Step 3: Data to vectors



#Each vector has as many dimensions as there are unique words in the SMS corpus

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])

print (len(bow_transformer.vocabulary_))

#bow_transformer.vocabulary_
# take one text message and get its bag-of-words counts as a vector, putting to use our new bow_transformer

message4 = messages['message'][3]

print (message4)

bow4 = bow_transformer.transform([message4])

print (bow4)
#nine unique words in message nr. 4, two of them appear twice, the rest only once. Sanity check: what are these words the appear twice?

print( bow_transformer.get_feature_names()[6736])

print( bow_transformer.get_feature_names()[8013])
#The bag-of-words counts for the entire SMS corpus are a large, sparse matrix:

messages_bow = bow_transformer.transform(messages['message'])

print ('sparse matrix shape:', messages_bow.shape)

print ('number of non-zeros:', messages_bow.nnz)

print ('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

#the term weighting and normalization can be done with TF-IDF, using scikit-learn's TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf4 = tfidf_transformer.transform(bow4)

print (tfidf4)
#IDF (inverse document frequency) of the word "u"? Of word "university"

print (tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])

print (tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])



messages_tfidf = tfidf_transformer.transform(messages_bow)

print (messages_tfidf.shape)
#Step 4: Training a model, detecting spam

#choosing the Naive Bayes classifier

spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])



print ('predicted:', spam_detector.predict(tfidf4)[0])

print ('expected:', messages.label[3])

all_predictions = spam_detector.predict(messages_tfidf)

print (all_predictions)



print( 'accuracy', accuracy_score(messages['label'], all_predictions))

print( 'confusion matrix\n', confusion_matrix(messages['label'], all_predictions))

print ('(row=expected, col=predicted)')
plt.matshow(confusion_matrix(messages['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')

plt.title('confusion matrix')

plt.colorbar()

plt.ylabel('expected label')

plt.xlabel('predicted label')





#this confusion matrix, we can compute precision and recall,

print (classification_report(messages['label'], all_predictions))


