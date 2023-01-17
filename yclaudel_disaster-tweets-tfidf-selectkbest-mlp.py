import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re

import string



plt.style.use('seaborn')

plt.rcParams['lines.linewidth'] = 1



from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.metrics import f1_score



NBR_STAR=70



X = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

y = X["target"]

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")



X.head()
X['keyword'].fillna('',inplace=True)

X['keyword'] = X['keyword'].map(lambda x:x.replace('%20', ' '))

test['keyword'].fillna('',inplace=True)

test['keyword'] = test['keyword'].map(lambda x:x.replace('%20', ' '))



# source https://www.kaggle.com/sahib12/nlp-starter-for-beginners

def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



#X['text'] = X['text'].apply(lambda x: clean_text(x))

#test['text'] = test['text'].apply(lambda x: clean_text(x))

# list of stop words

stop_word = list(ENGLISH_STOP_WORDS)

stop_word.append('http')

stop_word.append('https')

stop_word.append('û_')
X['target'].value_counts().plot(kind = 'barh')

plt.show()
def plot_sample_length_distribution(sample_texts):

    plt.figure(figsize=(10,10))

    plt.hist([len(s) for s in sample_texts], 50)

    plt.xlabel('Length of a sample')

    plt.ylabel('Number of samples')

    plt.title('Sample length distribution')

    plt.show()



plot_sample_length_distribution(X['text'])
keyword_stats = X.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(columns={'text':'Count', 'target':'Disaster Probability'})

keywords_disaster = keyword_stats.loc[keyword_stats['Disaster Probability']==1]

keywords_no_disaster  = keyword_stats.loc[keyword_stats['Disaster Probability']==0]

keyword_stats.sort_values('Disaster Probability', ascending=False).head(10)
from wordcloud import WordCloud, STOPWORDS



STOPWORDS.add('http')  

STOPWORDS.add('https')  

STOPWORDS.add('CO')  

STOPWORDS.add('û_')

no_disaster_text = " ".join(X[X["target"] == 0].text.to_numpy().tolist())

real_disaster_text = " ".join(X[X["target"] == 1].text.to_numpy().tolist())



no_disaster_cloud = WordCloud(stopwords=stop_word, background_color="white").generate(no_disaster_text)

real_disaster_cloud = WordCloud(stopwords=stop_word, background_color="white").generate(real_disaster_text)



def show_word_cloud(cloud, title):

  plt.figure(figsize = (16, 10))

  plt.imshow(cloud, interpolation='bilinear')

  plt.title(title)

  plt.axis("off")

  plt.show();



show_word_cloud(no_disaster_cloud, "No disaster common words")

show_word_cloud(real_disaster_cloud, "Real disaster common words")
vect = CountVectorizer(min_df=2,ngram_range=(1, 2), stop_words=stop_word)

X_train = vect.fit_transform(X['text'])

print("Vocabulary size: {}".format(len(vect.vocabulary_)))
all_ngrams = list(vect.get_feature_names())

num_ngrams = 50



all_counts = X_train.sum(axis=0).tolist()[0]

all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(

    zip(all_counts, all_ngrams), reverse=True)])

ngrams = list(all_ngrams)[:num_ngrams]

counts = list(all_counts)[:num_ngrams]



idx = np.arange(num_ngrams)

plt.figure(figsize=(10,10))

plt.barh(idx, counts,  color='orange')

plt.ylabel('N-grams')

plt.xlabel('Frequencies')

plt.title('Frequency distribution of n-grams')

plt.yticks(idx, ngrams)

plt.show()
# First try with this bag of word and a logistic regression



from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import ShuffleSplit



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=123)

scores = cross_val_score(LogisticRegression(), X_train, y,scoring="f1", cv=cv)

print("*"*NBR_STAR+"\n LogisticRegression on bag of word - cross-validation f1_score: {:.5f}\n".format(np.mean(scores))+"*"*NBR_STAR)
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(#max_df=0.1,         # drop words that occur in more than X percent of documents

                             min_df=10,      # only use words that appear at least X times

                             stop_words='english', # remove stop words

                             lowercase=True, # Convert everything to lower case 

                             use_idf=True,   # Use idf

                             norm=u'l2',     # Normalization

                             smooth_idf=True, # Prevents divide-by-zero errors

                             ngram_range=(1,3)

                            )

X_train = vect.fit_transform(X['text'])

# find maximum value for each of the features over dataset:

max_value = X_train.max(axis=0).toarray().ravel()

sorted_by_tfidf = max_value.argsort()

# get feature names

feature_names = np.array(vect.get_feature_names())



print("Features with lowest tfidf:\n{}".format(

      feature_names[sorted_by_tfidf[:20]]))



print("Features with highest tfidf: \n{}".format(

      feature_names[sorted_by_tfidf[-20:]]))
scores = cross_val_score(LogisticRegression(), X_train, y,scoring="f1", cv=cv)

print("*"*NBR_STAR+"\n LogisticRegression with tfidf - cross-validation f1_score: {:.5f}\n".format(np.mean(scores))+"*"*NBR_STAR)
from matplotlib.colors import ListedColormap, colorConverter, LinearSegmentedColormap



def visualize_coefficients(coefficients, feature_names, n_top_features=25):

    coefficients = coefficients.squeeze()

    if coefficients.ndim > 1:

        # this is not a row or column vector

        raise ValueError("coeffients must be 1d array or column vector, got"

                         " shape {}".format(coefficients.shape))

    coefficients = coefficients.ravel()



    if len(coefficients) != len(feature_names):

        raise ValueError("Number of coefficients {} doesn't match number of"

                         "feature names {}.".format(len(coefficients),

                                                    len(feature_names)))

    # get coefficients with large absolute values

    coef = coefficients.ravel()

    positive_coefficients = np.argsort(coef)[-n_top_features:]

    negative_coefficients = np.argsort(coef)[:n_top_features]

    interesting_coefficients = np.hstack([negative_coefficients,

                                          positive_coefficients])

    # plot them

    plt.figure(figsize=(20, 7))

    cm = ListedColormap(['#0000aa', '#ff2020'])

    colors = [cm(1) if c < 0 else cm(0)

              for c in coef[interesting_coefficients]]

    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients],

            color=colors)

    feature_names = np.array(feature_names)

    plt.subplots_adjust(bottom=0.3)

    plt.xticks(np.arange(1, 1 + 2 * n_top_features),

               feature_names[interesting_coefficients], rotation=60,

               ha="right")

    plt.ylabel("Coefficient magnitude")

    plt.xlabel("Words")

logreg = LogisticRegression()

logreg.fit(X_train, y)

y_predict = logreg.predict(X_train)

print("*"*NBR_STAR+"\n LogisticRegression with tfidf, no cross validation f1_score: {:.5f}\n".format(f1_score(y, y_predict, average='weighted'))+"*"*NBR_STAR)

visualize_coefficients(logreg.coef_, feature_names, n_top_features=50)
from tensorflow.python.keras import models

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras.layers import Dropout

import tensorflow as tf



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif
train_texts, val_texts, train_labels , val_labels = train_test_split(

    X['text'].values, X["target"].values, test_size=0.10, random_state=42)
vectorizer = TfidfVectorizer(

                             min_df=2,      # only use words that appear at least X times

                             #stop_words='english', # remove stop words

                             #lowercase=True, # Convert everything to lower case 

                             use_idf=True,   # Use idf

                             norm=u'l2',     # Normalization

                             smooth_idf=True, # Prevents divide-by-zero errors

                             ngram_range=(1,3),

                             #dtype='int32',

                             analyzer='word',

                             strip_accents = 'unicode',

                             decode_error = 'replace'

                            )

x_train = vectorizer.fit_transform(train_texts)

x_val = vectorizer.transform(val_texts)
selector = SelectKBest(f_classif, k=min(10000, x_train.shape[1]))

selector.fit(x_train, train_labels)

x_train = selector.transform(x_train)

x_val = selector.transform(x_val)



x_train = x_train.astype('float32')

x_val = x_val.astype('float32')
# model parameters

learning_rate=1e-4

epochs=1000

batch_size=128

layers=2

units=64

dropout_rate=0.2



model = models.Sequential()

model.add(Dropout(rate=dropout_rate, input_shape=x_train.shape[1:]))



for _ in range(layers-1):

    model.add(Dense(units=units, activation='relu'))

    model.add(Dropout(rate=dropout_rate))



model.add(Dense(units=1, activation='sigmoid'))
loss = 'binary_crossentropy'

optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])



# Create callback for early stopping on validation loss. If the loss does

# not decrease in two consecutive tries, stop training.

callbacks = [tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', patience=2)]



# Train and validate model.

history = model.fit(

        x_train.toarray(),

        train_labels,

        epochs=epochs,

        callbacks=callbacks,

        validation_data=(x_val.toarray(), val_labels),

        verbose=0,  # Logs once per epoch.

        batch_size=batch_size)



# Print results.

history = history.history

print('Validation accuracy: {acc}, loss: {loss}'.format(

        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
plt.plot(history['loss'])

plt.plot(history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
x_all = vectorizer.transform(X['text'].values)

x_all = selector.transform(x_all)

y_predict = model.predict_classes(x_all.toarray())

from sklearn.metrics import f1_score



score = f1_score(y, y_predict, average='weighted')

print("*"*NBR_STAR+"\n MLP Model f1_score: {:.5f}\n".format(score)+"*"*NBR_STAR)
y_predict[X.loc[X['keyword'].isin(list(keywords_disaster.index) )].index]=1

y_predict[X.loc[X['keyword'].isin(list(keywords_no_disaster.index) )].index]=0

score = f1_score(y, y_predict, average='weighted')

print("*"*NBR_STAR+"\n MLP Model f1_score: {:.5f}\n".format(score)+"*"*NBR_STAR)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

test_all = vectorizer.transform(test['text'].values)

test_all = selector.transform(test_all)



y_predict = model.predict_classes(test_all.toarray())

y_predict[test.loc[test['keyword'].isin(list(keywords_disaster.index) )].index]=1

y_predict[test.loc[test['keyword'].isin(list(keywords_no_disaster.index) )].index]=0



sample_submission["target"] = y_predict

sample_submission.to_csv("submission.csv", index=False)

sample_submission.head()