# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.datasets.base import get_data_home
%matplotlib inline

data_home = get_data_home()
twenty_home = os.path.join(data_home, "20news_home")

if not os.path.exists(data_home):
    os.makedirs(data_home)
    
if not os.path.exists(twenty_home):
    os.makedirs(twenty_home)
    
!cp ../input/20-newsgroup-sklearn/20news-bydate_py3* /tmp/scikit_learn_data
news_groups_train = fetch_20newsgroups(subset='train', shuffle=True, download_if_missing=False)
news_groups_test = fetch_20newsgroups(subset='test', shuffle=True, download_if_missing=False)
x_train, y_train = news_groups_train.data, news_groups_train.target
x_sp_train, x_sp_val, y_sp_train, y_sp_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
x_test, y_test = news_groups_test.data, news_groups_test.target
print(news_groups_train.target_names)
print("count for data in 20newsgroups", len(x_sp_train), len(x_sp_val), len(x_test))
print("count for train and validation data in 20newsgroups", len(x_sp_train) + len(x_sp_val), " and for test data", len(x_test))
print("count for train data in 20newsgroups", len(x_sp_train))
print("count for validation data in 20newsgroups", len(x_sp_val))
print("count for test data in 20newsgroups", len(x_test))
print(x_sp_train[10])
print(y_sp_train[10])
def show_distributation(data):
    dict = {}
    for index, name in enumerate(news_groups_train.target_names):
        dict.setdefault(name, np.sum(data==index))
    print(dict)
    print(dict.keys())
    print(dict.values())
    
    index = np.arange(len(news_groups_train.target_names))
    plt.figure(figsize=(10,5))
    plt.bar(index, dict.values())
    plt.xticks(index, dict.keys(), rotation=90)
    plt.title("category distributation")
    plt.xlabel("data count")
    plt.ylabel("data category")
    plt.show()
show_distributation(y_sp_train)
show_distributation(y_sp_val)
show_distributation(y_test)
def show_words(data):
    count = []
    for f in data:
        count.append(len(f.split()))
    plt.figure(figsize=(10,5))
    plt.hist(count, bins=20)
    plt.title("words distributation")
    plt.xlabel("words count")
    plt.ylabel("words weight")
    plt.show()
    
def show_chars(data):
    count = []
    for f in data:
        count.append(len(f))
    plt.figure(figsize=(10,5))
    plt.hist(count, bins=20)
    plt.title("chars distributation")
    plt.xlabel("chars count")
    plt.ylabel("chars count")
    plt.show()
show_words(x_sp_train)
show_chars(x_sp_train)
from sklearn.feature_extraction.text import CountVectorizer
from time import time
# refer: http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
# max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
# max_df = 25 means "ignore terms that appear in more than 25 documents".
# min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
# min_df = 5 means "ignore terms that appear in less than 5 documents".
    
vectorizer = CountVectorizer(max_df=0.97, min_df=3,
                                max_features=None,
                                stop_words='english')
t0 = time()
vec_x_train = vectorizer.fit_transform(x_sp_train)
print("done in %0.3fs." % (time() - t0))
print(vec_x_train)
from sklearn.feature_extraction.text import TfidfVectorizer
# refer: http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
tfidf_vectorizer = TfidfVectorizer(max_df=0.97, min_df=2,
                                max_features=None,
                                stop_words='english')
t0 = time()
tfidf_x_sp_train = tfidf_vectorizer.fit_transform(x_sp_train)
tfidf_x_sp_val = tfidf_vectorizer.transform(x_sp_val)

tfidf_x_train = tfidf_vectorizer.transform(x_train)
tfidf_x_test = tfidf_vectorizer.transform(x_test)
print("done in %0.3fs." % (time() - t0))
print(tfidf_x_sp_train.shape)
print(tfidf_x_sp_val.shape)
print(tfidf_x_train.shape)
print(tfidf_x_test.shape)
print(tfidf_x_sp_train)
# refer: https://radimrehurek.com/gensim/models/word2vec.html
# https://www.programcreek.com/python/example/98848/gensim.models.word2vec.Word2Vec
# http://mattmahoney.net/dc/text8.zip
import os
import gensim
from gensim.models import word2vec
EMBEDDING_DIM = 100
# if not os.path.isfile('data/w2c/word2vec_model.model'):
#     sentences = word2vec.Text8Corpus('data/w2c/text8')
#     word2vec_model = word2vec.Word2Vec(sentences, size=EMBEDDING_DIM, min_count=1, sg=0)
#     word2vec_model.save('data/w2c/word2vec_model.model')
#     print("word2vec_model is saved")
word2vec_model = word2vec.Word2Vec.load('../input/word2vec-modelmodel/word2vec_model.model')
print("word2vec_model is loaded")
from sklearn.neural_network import MLPClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def show_performance(model, x_train, y_train, x_val, y_val):    
    results = {}
    results['model_name'] = model.__class__.__name__
    t0 = time()
    model.fit(x_train, y_train)
    results['train_time'] = time() - t0
    t1 = time()
    predicts = model.predict(x_train)
    results['val_time'] = time() - t1
    train_score = model.score(x_train, y_train)
    val_score = model.score(x_val, y_val)
    results['train_score'] = train_score
    results['val_score'] = val_score
    print(results)
    
lr = LogisticRegression(C=1.0, penalty='l2')
show_performance(lr, tfidf_x_sp_train, y_sp_train, tfidf_x_sp_val, y_sp_val)
svc = SVC(kernel='linear', C=0.5, gamma=0.9, random_state=0)
show_performance(svc, tfidf_x_sp_train, y_sp_train, tfidf_x_sp_val, y_sp_val)
gnb = MultinomialNB(alpha=0.5)
show_performance(gnb, tfidf_x_sp_train, y_sp_train, tfidf_x_sp_val, y_sp_val)
#refer: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from multiprocessing import cpu_count
from sklearn import feature_selection

fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=40)
def show_performance_with_gscv(name, model, x_train, y_train, params):
    results = {}
    results['model_name'] = name
    gscv = GridSearchCV(model, params, cv=3, n_jobs=cpu_count()-1, return_train_score=True)
#     x_train_fs = fs.fit_transform(x_train, y_train)
    
    gscv.fit(x_train, y_train)
    
    results['params'] = gscv.best_params_
    results['train_time'] = np.mean(gscv.cv_results_['mean_fit_time'])
    results['val_time'] = np.mean(gscv.cv_results_['mean_score_time'])
    results['train_score'] = gscv.cv_results_['mean_train_score'][gscv.best_index_]
    
    # it is get fro train data set, could be taken as a val result
    results['val_score'] = gscv.cv_results_['mean_test_score'][gscv.best_index_]
    
    results['best_model'] = gscv.best_estimator_
    
    return results
params = {'C': [0.01, 1, 3]}
rs_lr = show_performance_with_gscv('LogisticRegression', LogisticRegression(penalty='l2'), tfidf_x_train, y_train, params)
print(rs_lr)
params = {'C': [0.1, 1, 3], 'gamma':[0.5, 0.9]}
rs_svc = show_performance_with_gscv('SVC', SVC(kernel='linear'), tfidf_x_train, y_train, params)
print(rs_svc)
params = {'alpha': [0.0001, 0.01, 0.5, 0.95]}
rs_nb = show_performance_with_gscv('NaiveBayes', MultinomialNB(), tfidf_x_train, y_train, params)
print(rs_nb)
def autolabel(ax, rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')
            
def show_metrics(rs_lr, rs_svc, rs_nb):
#     https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

    train_time = (rs_lr['train_time'], rs_svc['train_time'], rs_nb['train_time'])
    val_time = (rs_lr['val_time'], rs_svc['val_time'], rs_nb['val_time'])
    
    train_score = (rs_lr['train_score'], rs_svc['train_score'], rs_nb['train_score'])
    val_score = (rs_lr['val_score'], rs_svc['val_score'], rs_nb['val_score'])

    ind = np.arange(len(train_time))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax0 = plt.subplots(1, 1, figsize = (16,5))
    
    rects1 = ax0.bar(ind - width/2, train_time, width,
                    color='SkyBlue', label='train time')
    rects2 = ax0.bar(ind + width/2, val_time, width,
                    color='IndianRed', label='val time')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax0.set_ylabel('time')
    ax0.set_title('time by train and val')
    ax0.set_xticks(ind)
    ax0.set_xticklabels(('LogisticRegression', 'SVC', 'NaiveBayes'))
    ax0.legend()
    autolabel(ax0, rects1, "left")
    autolabel(ax0, rects2, "right")
    plt.show()
    
    fig, ax1 = plt.subplots(1, 1, figsize = (16,5))
    
    rects3 = ax1.bar(ind - width/2, train_score, width,
                    color='SkyBlue', label='train score')
    rects4 = ax1.bar(ind + width/2, val_score, width,
                    color='IndianRed', label='val score')
    ax1.set_ylabel('Scores')
    ax1.set_title('Scores by train and val')
    ax1.set_xticks(ind)
    ax1.set_xticklabels(('LogisticRegression', 'SVC', 'NaiveBayes'))
    ax1.legend()
    autolabel(ax1, rects3, "left")
    autolabel(ax1, rects4, "right")
    plt.show()
    
show_metrics(rs_lr, rs_svc, rs_nb)
# tfidf_x_test_fs = fs.transform(tfidf_x_test)
rs_lr_acc = rs_lr['best_model'].score(tfidf_x_test, y_test)
rs_svc_acc = rs_svc['best_model'].score(tfidf_x_test, y_test)
rs_nb_acc = rs_nb['best_model'].score(tfidf_x_test, y_test)
print("LogisticRegression best score", rs_lr_acc)
print("svc best score", rs_svc_acc)
print("NaiveBayes best score", rs_nb_acc)
#https://keras.io/getting-started/sequential-model-guide/
#https://github.com/dennybritz/cnn-text-classification-tf
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Reshape, concatenate, Dropout
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Embedding
from keras import optimizers
from keras.callbacks import EarlyStopping
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
words_count = []
for i in x_train:
    words_count.append(len(text_to_word_sequence(i, split=' ')))
print(np.max(words_count))
print(np.min(words_count))
# 18~ 16333
NUM_WORDS = 20000
MAX_LEN= 1000
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(x_sp_train)
word_index = tokenizer.word_index
x_sp_train_dl = pad_sequences(tokenizer.texts_to_sequences(x_sp_train), maxlen=MAX_LEN)
y_sp_train_dl = to_categorical(np.asarray(y_sp_train), num_classes=20)

x_sp_val_dl = pad_sequences(tokenizer.texts_to_sequences(x_sp_val), maxlen=MAX_LEN)
y_sp_val_dl = to_categorical(np.asarray(y_sp_val), num_classes=20)

x_test_dl = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=MAX_LEN)
y_test_dl = to_categorical(np.asarray(y_test), num_classes=20)
def text_CNN(embedding_layer):
    sequence_input = Input(shape=(MAX_LEN,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
   
    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    
    embedded_sequences = Reshape((MAX_LEN, EMBEDDING_DIM, 1))(embedded_sequences)
    x = Conv2D(100, (5, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    x = MaxPooling2D((MAX_LEN - 5 + 1, 1))(x)

    y = Conv2D(100, (4, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    y = MaxPooling2D((MAX_LEN - 4 + 1, 1))(y)

    z = Conv2D(100, (3, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    z = MaxPooling2D((MAX_LEN - 3 + 1, 1))(z)

    alpha = concatenate([x,y,z])
    alpha = Flatten()(alpha)
    alpha = Dropout(0.5)(alpha)
    preds = Dense(len(news_groups_train.target_names), activation='softmax')(alpha)
    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta()
        
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])
    return model
def show_history(history):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
min_num_words = min(NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((min_num_words+1, EMBEDDING_DIM))

for word, index in word_index.items():
    if index > min_num_words:
        continue
    elif word in word2vec_model:
            embedding_matrix[index] = word2vec_model[word]

print('embedding matrix shape: {}'.format(embedding_matrix.shape))   
embedding_layer = Embedding(min_num_words+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_LEN, 
                            trainable=False)
text_cnn = text_CNN(embedding_layer)
text_cnn.summary()
# https://keras.io/getting-started/sequential-model-guide/#training
early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')

history = text_cnn.fit(x_sp_train_dl, y_sp_train_dl, validation_data=(x_sp_val_dl, y_sp_val_dl), epochs=60, batch_size=50,callbacks=[early_stopping])
show_history(history.history)
rs_text_cnn = text_cnn.evaluate(x_test_dl, y_test_dl, batch_size=50)
rs_text_cnn_acc = rs_text_cnn[1]
train_accs = (rs_lr['train_score'], rs_svc['train_score'], rs_nb['train_score'], history.history['acc'][-1])
val_accs = (rs_lr['val_score'], rs_svc['val_score'], rs_nb['val_score'], history.history['val_acc'][-1])
test_accs = (rs_lr_acc, rs_svc_acc, rs_nb_acc, rs_text_cnn_acc)
ind = np.arange(len(train_accs))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize = (16,5))

rects0 = ax.bar(ind-width/3, train_accs, width/3,
                color='Blue', label='train acc')
rects1 = ax.bar(ind, val_accs, width/3,
                color='SkyBlue', label='val acc')
rects2 = ax.bar(ind+width/3, test_accs, width/3,
                color='Green', label='test acc')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('acc')
ax.set_xlabel('model')
ax.set_title('acc for each model')
ax.set_xticks(ind)
ax.set_xticklabels(('LogisticRegression', 'SVC', 'NaiveBayes', 'text_cnn'))
ax.legend()
autolabel(ax, rects0, 'left')
autolabel(ax, rects1, 'center')
autolabel(ax, rects2, 'right')
plt.show()

plt.show()
