import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from nltk.tokenize import word_tokenize, sent_tokenize

import nltk
import pandas as pd

df = pd.read_csv('/kaggle/input/traveldataset5000/5000TravelQuestionsDataset.csv', encoding="Latin-1", header=None)
df[1].unique()
df[1].replace({'TGU\n':'TGU', 'TTD\n':'TTD' , '\nENT': 'ENT'}, inplace=True)

print(df[1].unique())

print(len(df[1].unique()))
df[2].unique()
df[2].replace({'WTHTMP\n':'WTHTMP', '\nTGULAU': 'TGULAU', 'TRSOTH\n': 'TRSOTH', 'FODBAK\n': 'FODBAK', 'TRSAIR\n':'TRSAIR', 'TGUCIG\n':'TGUCIG', 'TTDOTH\n':'TTDOTH', 'WTHOTH\n':'WTHOTH', 'TTDSIG\n':'TTDSIG', 

              'TGUOTH\n':'TGUOTH', 'TTDSHP\n':'TTDSHP', 'TRSROU\n':'TRSROU', 'TTDSPO\n':'TTDSPO', '\nACMOTH':'ACMOTH', 'ACMOTH\n':'ACMOTH', '\nWTHOTH':'WTHOTH' }, inplace=True)

print(df[2].unique())

print(len(df[2].unique()))
travelQsns = df[0]

print(travelQsns[0:10])
regexTokenizer = nltk.RegexpTokenizer(r"[a-zA-Z0-9]+")



# travelTokens = [word_tokenize(qsn.lower()) for qsn in df[0]]

travelTokens = [regexTokenizer.tokenize(qsn.lower()) for qsn in df[0]]

print(travelTokens[:10])
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords



stop_words = set(stopwords.words('english'))



travelTokensLemmatized=[]

for sent in travelTokens:

    a=list()

    for w in sent:

        w = wordnet_lemmatizer.lemmatize(w)

        if w not in stop_words:

            a.append(w)

    travelTokensLemmatized.append(a)
print(travelTokensLemmatized[:10])
from nltk.tokenize.treebank import TreebankWordDetokenizer

ppUntokenizedQsns = [TreebankWordDetokenizer().detokenize(qsn) for qsn in travelTokensLemmatized]

print(ppUntokenizedQsns[:10])
from sklearn.feature_extraction.text import TfidfVectorizer



tfidfVectorizerUnigram = TfidfVectorizer()

unigramTfidfVectors = tfidfVectorizerUnigram.fit_transform(ppUntokenizedQsns)



print(unigramTfidfVectors[:10])
unigramTfidfFeature_names = ['tfidf_'+ fn for fn in tfidfVectorizerUnigram.get_feature_names()]

print(unigramTfidfFeature_names[:10])
tfidfFeaturesDf = pd.DataFrame.sparse.from_spmatrix(unigramTfidfVectors)

tfidfFeaturesDf.columns = unigramTfidfFeature_names

tfidfFeaturesDf.head()
from nltk.util import ngrams



bigramAllQsns = []



for qsn in travelTokensLemmatized:

    n_grams  = ngrams(qsn, 2)

    bigramAllQsns.append([ '_'.join(grams) for grams in n_grams])



print(bigramAllQsns[:10])
untokenizedBigrams = [TreebankWordDetokenizer().detokenize(qsn) for qsn in bigramAllQsns]

tfidfVectorizer = TfidfVectorizer()

bigramVectors = tfidfVectorizer.fit_transform(untokenizedBigrams)

bigramFeature_names = ['bigram_'+ fn for fn in tfidfVectorizer.get_feature_names()]

print(bigramVectors.shape)



bigramFeaturesDf = pd.DataFrame.sparse.from_spmatrix(bigramVectors)

bigramFeaturesDf.columns = bigramFeature_names

bigramFeaturesDf.head()
posTagsAllQsns = list()



for qsn in travelTokensLemmatized:

    posTagsAllQsns.append(nltk.pos_tag(qsn))        
posTagFormatted = []

for qsn in posTagsAllQsns:

    posTagSent = []

    for w,pos in qsn:

        posTagSent.append(w + "_" +pos)

    posTagFormatted.append(posTagSent)
print(posTagFormatted[:10])
untokenizedPosTags = [TreebankWordDetokenizer().detokenize(qsn) for qsn in posTagFormatted]

tfidfVectorizer = TfidfVectorizer()

posTagVectors = tfidfVectorizer.fit_transform(untokenizedPosTags)

posTagFeature_names = ['pos_'+ fn for fn in tfidfVectorizer.get_feature_names()]

print(posTagVectors.shape)



posTagFeaturesDf = pd.DataFrame.sparse.from_spmatrix(posTagVectors)

posTagFeaturesDf.columns = posTagFeature_names

posTagFeaturesDf.head()
import spacy

spacy_nlp = spacy.load('en_core_web_sm')
nerList = []

for qsn in travelQsns:

    doc = spacy_nlp(qsn)

    nerPerQsn = []

    for i in doc.ents:

        nerPerQsn.append((i.lemma_).lower() + "_" + i.label_)

    nerList.append(nerPerQsn)



print(nerList[:10])
untokenizedNERList = [TreebankWordDetokenizer().detokenize(qsn) for qsn in nerList]
print(untokenizedNERList[:10])
tfidfVectorizer = TfidfVectorizer()

nerVectors = tfidfVectorizer.fit_transform(untokenizedNERList)

nerFeature_names = ['ner_'+ fn for fn in tfidfVectorizer.get_feature_names()]

print(nerVectors.shape)



nerFeaturesDf = pd.DataFrame.sparse.from_spmatrix(nerVectors)

nerFeaturesDf.columns = nerFeature_names

nerFeaturesDf.head()
ppUntokenizedQsns[:10]
headwordPairListsAllQsns = []

for qsn in travelQsns:

    doc = spacy_nlp(qsn)

    headWordPairPerQsn = []

    for token in doc:

        if (token.text.lower() not in stop_words) and (token.head.text.lower() not in stop_words):

            headWordPairPerQsn.append(token.text.lower() + "_" + token.head.text.lower())

    headwordPairListsAllQsns.append(headWordPairPerQsn)



print(headwordPairListsAllQsns[:10])
untokenizedHeadWordPairListsAllQsns = [TreebankWordDetokenizer().detokenize(qsn) for qsn in headwordPairListsAllQsns]

print(untokenizedHeadWordPairListsAllQsns[:10])
tfidfVectorizer = TfidfVectorizer()

wordHeadWordPairVectors = tfidfVectorizer.fit_transform(untokenizedHeadWordPairListsAllQsns)

wordHeadWordPairFeature_names = ['whwpair_'+ fn for fn in tfidfVectorizer.get_feature_names()]

print(wordHeadWordPairVectors.shape)



wordHeadWordPairFeaturesDf = pd.DataFrame.sparse.from_spmatrix(wordHeadWordPairVectors)

wordHeadWordPairFeaturesDf.columns = wordHeadWordPairFeature_names

wordHeadWordPairFeaturesDf.head()
finalFeatureDf = pd.concat([df[0],tfidfFeaturesDf,bigramFeaturesDf,posTagFeaturesDf,nerFeaturesDf,wordHeadWordPairFeaturesDf,df[1], df[2]], axis=1)

finalFeatureDf.head()

print(finalFeatureDf.columns)
X = finalFeatureDf.drop([0,1,2], axis=1)
# y_coarse_class = pd.DataFrame(data=finalFeatureDf[1])

# y_fine_class = pd.DataFrame(data=finalFeatureDf[2])



y_coarse_class = pd.DataFrame(data=df[1])

y_fine_class = pd.DataFrame(data=df[2])
from sklearn import preprocessing



le_coarse_class = preprocessing.LabelEncoder()

le_coarse_class.fit(y_coarse_class.iloc[:,-1])

print(le_coarse_class.classes_)
y_coarse_classEncoded = pd.DataFrame(data=le_coarse_class.transform(y_coarse_class.iloc[:,-1]))

y_coarse_classEncoded.head()
le_fine_class = preprocessing.LabelEncoder()

le_fine_class.fit(y_fine_class.iloc[:,-1])

print(le_fine_class.classes_)



y_fine_classEncoded = pd.DataFrame(data=le_fine_class.transform(y_fine_class.iloc[:,-1]))

y_fine_classEncoded.head()
y_coarse_classEncoded.columns = ['coarse_class']

y_fine_classEncoded.columns = ['fine_class']
labelEncodedFullDf = pd.concat([X, y_coarse_classEncoded, y_fine_classEncoded], axis=1)

labelEncodedFullDf.head()
X.head()
y_coarse_classEncoded.head()
from sklearn.model_selection import KFold

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_validate
print(X.shape)
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



X_features = X.to_numpy()

Y_coarse_class = np.array(y_coarse_classEncoded.iloc[:,-1])

coarseLabels = np.unique(Y_coarse_class)
parameters = {'alpha':[0.1,0.3],'loss':['hinge'],'max_iter':[20,30]}



classifier = SGDClassifier(random_state=42, tol=None)

grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy', verbose=1)

grid.fit(X_features[:2000], Y_coarse_class[:2000])



print('======Best Parameters=====')

print(grid.best_params_)

print(grid.best_score_)
kf = StratifiedKFold(n_splits=10)



svm_conf_mat = []

svm_clas_repo = []

fold = 0



acc_coarse=[]

prec_coarse=[]

recall_coarse=[]

f1_coarse=[]



for train_index, test_index in kf.split(X_features,Y_coarse_class):

    fold += 1

    print("*"*50 + "Fold = " + str(fold))

    

    X_train, X_test = X_features[train_index], X_features[test_index]

    y_train, y_test = Y_coarse_class[train_index], Y_coarse_class[test_index]

    

    svmclassifier = SGDClassifier(loss='log', penalty='l2',alpha=0.01, random_state=13, max_iter=20, tol=None)

    svmclassifier.fit(X_train, y_train)

    

    y_pred = svmclassifier.predict(X_test)

    

    conf = confusion_matrix(y_test, y_pred, labels=coarseLabels)

    svm_conf_mat.append(conf)

    

    clfReport = classification_report(y_test, y_pred)

    svm_clas_repo.append(clfReport)

    

    print(conf)

    print(clfReport)

    

    accuracy = accuracy_score(y_test, y_pred)

    acc_coarse.append(accuracy)

    print('Accuracy: %f' % accuracy)

    

    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)

    print('Precision: %f' % precision)

    prec_coarse.append(precision)

    

    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)

    print('Recall: %f' % recall)

    recall_coarse.append(recall)

    

    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

    print('F1 score: %f' % f1)

    f1_coarse.append(f1)
svm_conf_mat=np.array(svm_conf_mat)

svm_clas_repo=np.array(svm_clas_repo)



print("Coarse class mean accuracy={}".format(np.mean(acc_coarse)))

print("Coarse class mean precision={}".format(np.mean(prec_coarse)))

print("Coarse class mean recall={}".format(np.mean(recall_coarse)))

print("Coarse class mean f1={}".format(np.mean(f1_coarse)))
sumMat = np.zeros((7,7))

for mat in svm_conf_mat:

    sumMat = np.add(sumMat,mat)



sumMat = pd.DataFrame(sumMat, index=coarseLabels, columns=coarseLabels)

print("Cofusion matrix over 10-fold cross validation")

print(sumMat)
Y_fine_class = np.array(y_fine_classEncoded.iloc[:,-1])

fineLabels = np.unique(Y_fine_class)
parameters = {'alpha':[1e-1,1e-2],'loss':['hinge','log'],'max_iter':[10,20]}



classifier = SGDClassifier(random_state=42, tol=None)

grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy', verbose=1)

grid.fit(X_features[:1000], Y_fine_class[:1000])



print('======Best Parameters=====')

print(grid.best_params_)

print(grid.best_score_)
kf = StratifiedKFold(n_splits=10)



svm_conf_mat_fine = []

svm_clas_report_fine = []

fold = 0



acc_fine=[]

prec_fine=[]

recall_fine=[]

f1_fine=[]



for train_index, test_index in kf.split(X_features, Y_fine_class):

    fold += 1

    print("*"*50 + "Fold = " + str(fold))

    

    X_train, X_test = X_features[train_index], X_features[test_index]

    y_train, y_test = Y_fine_class[train_index], Y_fine_class[test_index]

    

    svmclassifier = SGDClassifier(loss='hinge', penalty='l2',alpha=0.01, random_state=13, max_iter=15, tol=None)

    svmclassifier.fit(X_train, y_train)

    

    y_pred = svmclassifier.predict(X_test)

    

    conf = confusion_matrix(y_test, y_pred, labels=fineLabels)

    svm_conf_mat_fine.append(conf)

    

    clfReport = classification_report(y_test, y_pred)

    svm_clas_report_fine.append(clfReport)

    

    accuracy = accuracy_score(y_test, y_pred)

    acc_fine.append(accuracy)

    print('Accuracy: %f' % accuracy)

    

    precision = precision_score(y_test, y_pred,average='macro', zero_division=1)

    print('Precision: %f' % precision)

    prec_fine.append(precision)

    

    recall = recall_score(y_test, y_pred,average='macro', zero_division=1)

    print('Recall: %f' % recall)

    recall_fine.append(recall)

    

    f1 = f1_score(y_test, y_pred,average='macro', zero_division=1)

    print('F1 score: %f' % f1)

    f1_fine.append(f1)



    

print("Fine class mean accuracy={}".format(np.mean(acc_fine)))

print("Fine class mean precision={}".format(np.mean(prec_fine)))

print("Fine class mean recall={}".format(np.mean(recall_fine)))

print("Fine class mean f1={}".format(np.mean(f1_fine)))
sumMat = np.zeros((len(fineLabels),len(fineLabels)))

for mat in svm_conf_mat_fine:

    sumMat = np.add(sumMat,mat)



sumMat = pd.DataFrame(sumMat, index=fineLabels, columns=fineLabels)



print("Fine Class confusion matrix over 10-fold cross validation")

pd.set_option('display.max_columns', 63)

print(sumMat.describe())
print(travelTokensLemmatized[:10])
from gensim.models import Word2Vec



skipgramModel = Word2Vec(min_count=1, 

                         sg=1, 

                         size=300,

                         window=5,

                        seed=3)



skipgramModel.build_vocab(sentences = travelTokensLemmatized, 

                           progress_per=1000)



skipgramModel.train(sentences = travelTokensLemmatized,

                     total_examples=skipgramModel.corpus_count, 

                     epochs=10, 

                     report_delay=1)
sentIndex = 0

sentVectors = []



for sent in travelTokensLemmatized:

    sentVec=np.zeros((300,))

    for w in sent:

        wordVector = skipgramModel.wv[w]

        sentVec += wordVector

    sentVectors.append(sentVec)



print(sentVectors[:5])
sentVectorsDf = pd.DataFrame(sentVectors)

sentVectorsDf.describe()
wordEmbeddingDf = pd.concat([df[0], sentVectorsDf, y_coarse_classEncoded, y_fine_classEncoded], axis=1)

wordEmbeddingDf.head()
X_wordEmbed = wordEmbeddingDf.iloc[:,1: -2]

X_wordEmbed =  X_wordEmbed.to_numpy()



Y_coarse_class = np.array(y_coarse_classEncoded.iloc[:,-1])

coarseLabels = np.unique(Y_coarse_class)
parameters = {'alpha':[0.1,0.3],'loss':['hinge','log'],'max_iter':[10,15]}



classifier = SGDClassifier(random_state=42, tol=None)

grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy', verbose=1)

grid.fit(X_wordEmbed[:2000], Y_coarse_class[:2000])



print('======Best Parameters=====')

print(grid.best_params_)

print(grid.best_score_)
kf = StratifiedKFold(n_splits=10)



svm_conf_mat = []

svm_clas_repo = []

fold = 0



acc_coarse=[]

prec_coarse=[]

recall_coarse=[]

f1_coarse=[]



for train_index, test_index in kf.split(X_wordEmbed, Y_coarse_class):

    fold += 1

    print("*"*50 + "Fold = " + str(fold))

    

    X_train, X_test = X_wordEmbed[train_index], X_wordEmbed[test_index]

    y_train, y_test = Y_coarse_class[train_index], Y_coarse_class[test_index]

    

    svmclassifier = SGDClassifier(max_iter=10, tol=1e-3, loss='hinge', penalty='l2', alpha=0.1, random_state=42)

    svmclassifier.fit(X_train, y_train)

    

    y_pred = svmclassifier.predict(X_test)

    

    conf = confusion_matrix(y_test, y_pred, labels=coarseLabels)

    svm_conf_mat.append(conf)

    

    clfReport = classification_report(y_test, y_pred)

    svm_clas_repo.append(clfReport)

    

    print(conf)

    print(clfReport)

    

    accuracy = accuracy_score(y_test, y_pred)

    acc_coarse.append(accuracy)

    print('Accuracy: %f' % accuracy)

    

    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)

    print('Precision: %f' % precision)

    prec_coarse.append(precision)

    

    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)

    print('Recall: %f' % recall)

    recall_coarse.append(recall)

    

    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

    print('F1 score: %f' % f1)

    f1_coarse.append(f1)

    
svm_conf_mat=np.array(svm_conf_mat)

svm_clas_repo=np.array(svm_clas_repo)



print("Coarse class mean accuracy={}".format(np.mean(acc_coarse)))

print("Coarse class mean precision={}".format(np.mean(prec_coarse)))

print("Coarse class mean recall={}".format(np.mean(recall_coarse)))

print("Coarse class mean f1={}".format(np.mean(f1_coarse)))



sumMat = np.zeros((7,7))

for mat in svm_conf_mat:

    sumMat = np.add(sumMat,mat)



sumMat = pd.DataFrame(sumMat, index=coarseLabels, columns=coarseLabels)

print("WordEmbedding Confusion matrix for coarse class over 10-fold cross validation")

sumMat
parameters = {'alpha':[0.01,0.001], 'max_iter':[20,80]}



classifier = SGDClassifier(random_state=42, tol=None, loss='log')

grid = GridSearchCV(classifier, parameters, cv = 10,  scoring = 'accuracy', verbose=1)

grid.fit(X_wordEmbed[:2000], Y_fine_class[:2000])



print('======Best Parameters=====')

print(grid.best_params_)

print(grid.best_score_)
Y_fine_class = np.array(y_fine_classEncoded.iloc[:,-1])

fineLabels = np.unique(Y_fine_class)



kf = StratifiedKFold(n_splits=10)



svm_conf_mat_fine = []

svm_clas_report_fine = []

fold = 0



acc_fine=[]

prec_fine=[]

recall_fine=[]

f1_fine=[]



for train_index, test_index in kf.split(X_wordEmbed, Y_fine_class):

    fold += 1

    print("*"*50 + "Fold = " + str(fold))

    

    X_train, X_test = X_wordEmbed[train_index], X_wordEmbed[test_index]

    y_train, y_test = Y_fine_class[train_index], Y_fine_class[test_index]

    

    svmclassifier = SGDClassifier(max_iter=90, tol=1e-3, loss='log', penalty='l2', alpha=0.001, random_state=42)

    svmclassifier.fit(X_train, y_train)

    

    y_pred = svmclassifier.predict(X_test)

    

    conf = confusion_matrix(y_test, y_pred, labels=fineLabels)

    svm_conf_mat_fine.append(conf)

    

    clfReport = classification_report(y_test, y_pred)

    svm_clas_report_fine.append(clfReport)

    

    accuracy = accuracy_score(y_test, y_pred)

    acc_fine.append(accuracy)

    print('Accuracy: %f' % accuracy)

    

    precision = precision_score(y_test, y_pred,average='macro', zero_division=1)

    print('Precision: %f' % precision)

    prec_fine.append(precision)

    

    recall = recall_score(y_test, y_pred,average='macro', zero_division=1)

    print('Recall: %f' % recall)

    recall_fine.append(recall)

    

    f1 = f1_score(y_test, y_pred,average='macro', zero_division=1)

    print('F1 score: %f' % f1)

    f1_fine.append(f1)
print("Fine class mean accuracy={}".format(np.mean(acc_fine)))

print("Fine class mean precision={}".format(np.mean(prec_fine)))

print("Fine class mean recall={}".format(np.mean(recall_fine)))

print("Fine class mean f1={}".format(np.mean(f1_fine)))



sumMat = np.zeros((len(fineLabels),len(fineLabels)))

for mat in svm_conf_mat_fine:

    sumMat = np.add(sumMat,mat)



sumMat = pd.DataFrame(sumMat, index=fineLabels, columns=fineLabels)



print("Word embedding fine grained Class confusion matrix over 10-fold cross validation")

sumMat
import tensorflow as tf

from scipy.sparse import hstack

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer 

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from gensim.models import Word2Vec

from gensim.models.keyedvectors import KeyedVectors

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.callbacks import EarlyStopping
from numpy import asarray

embeddings_index = dict()

f = open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt')

for line in f:

    values = line.split()

    word = values[0]

    coefs = asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
word_tokenizer = Tokenizer()

word_tokenizer.fit_on_texts(ppUntokenizedQsns)

word_index = word_tokenizer.word_index



vocab_size = len(word_tokenizer.word_index) + 1



weight_matrix = np.zeros((vocab_size, 100))



for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        weight_matrix[i] = embedding_vector
preprocessed_questions_sequences = word_tokenizer.texts_to_sequences(ppUntokenizedQsns)

preprocessed_questions_sequences_padded = pad_sequences(preprocessed_questions_sequences, maxlen=100, padding='post', truncating='post')
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=19)

cvscores = []

iteration = 0



cv_accuracy_scores = []

cv_conf_matrices = []

cv_recall_scores = []

cv_precision_scores = []

cv_f1_scores = []

lstm_conf_mat_coarse = []



questions_list = np.array(preprocessed_questions_sequences_padded.tolist());

category_list = np.array(Y_coarse_class);



for train_index, test_index in kfold.split(questions_list,category_list):



    iteration = iteration + 1

    print("Fold =================================================================== ", iteration)



    train_questions,test_questions=questions_list[train_index],questions_list[test_index]

    train_category_labels,test_category_labels=category_list[train_index],category_list[test_index]

    

    model = tf.keras.Sequential([

    tf.keras.layers.Embedding(input_dim=len(word_index)+1, input_length=100, output_dim=100, weights=[weight_matrix], trainable=False),

    tf.keras.layers.SpatialDropout1D(0.3),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200)),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(7, activation='softmax')

    ])

 

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    earlystopCallback = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')

    history = model.fit(train_questions, train_category_labels, epochs=50, batch_size=500, verbose=1, callbacks=[earlystopCallback])

    

    print(history.history['loss'])

    print(history.history['accuracy'])



    # Evaluate the model

    predicted_category_labels = model.predict_classes(test_questions, verbose=0)  

    

    conf = confusion_matrix(test_category_labels, predicted_category_labels, labels=coarseLabels)

    lstm_conf_mat_coarse.append(conf)

    

    accuracy = accuracy_score(test_category_labels, predicted_category_labels)

    recall = recall_score(test_category_labels, predicted_category_labels,average=None, zero_division = 0)

    precision = precision_score(test_category_labels, predicted_category_labels,average=None, zero_division = 0)

    f1 = f1_score(test_category_labels, predicted_category_labels,average=None, zero_division = 0)

    

    cv_accuracy_scores.append(accuracy)

    cv_recall_scores.append(recall)

    cv_precision_scores.append(precision)

    cv_f1_scores.append(f1)
print("LSTM Accuracy={}".format(np.mean(cv_accuracy_scores)))
print("LSTM reacll={}".format(np.mean(cv_recall_scores)))
print("LSTM precision={}".format(np.mean(cv_precision_scores)))
print("LSTM f1={}".format(np.mean(cv_f1_scores)))
sumMat = np.zeros((len(coarseLabels),len(coarseLabels)))

for mat in lstm_conf_mat_coarse:

    sumMat = np.add(sumMat,mat)



sumMat = pd.DataFrame(sumMat, index=coarseLabels, columns=coarseLabels)



print("LSTM coarse class confusion matrix over 10-fold cross validation")

sumMat
kfold = StratifiedKFold(n_splits=10, shuffle=True)

cvscores = []

iteration = 0



cv_accuracy_scores = []

cv_conf_matrices = []

cv_recall_scores = []

cv_precision_scores = []

cv_f1_scores = []

lstm_conf_mat_fine = []



preprocessed_qsns = np.array(preprocessed_questions_sequences_padded.tolist());

fine_class_labels = np.array(Y_fine_class);



for train_index, test_index in kfold.split( preprocessed_qsns,fine_class_labels):



    iteration = iteration + 1

    print("Fold =================================================================== ", iteration)



    train_questions,test_questions= preprocessed_qsns[train_index], preprocessed_qsns[test_index]

    train_labels,test_labels=fine_class_labels[train_index],fine_class_labels[test_index]

    

    model = tf.keras.Sequential([

    tf.keras.layers.Embedding(input_dim=len(word_index)+1, input_length=100, output_dim=100, weights=[weight_matrix], trainable=False),

    tf.keras.layers.SpatialDropout1D(0.3),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200)),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(63, activation='softmax')

    ])

 

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    earlystopCallback = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')

    history = model.fit(train_questions, train_labels, epochs=80, batch_size=500, verbose=1, callbacks=[earlystopCallback])

    

    print(history.history['loss'])

    print(history.history['accuracy'])



    # Evaluate the model

    predicted_category_labels = model.predict_classes(test_questions, verbose=0)  

    

    conf = confusion_matrix(test_labels, predicted_category_labels, labels=fineLabels)

    lstm_conf_mat_fine.append(conf)

    

    accuracy = accuracy_score(test_labels, predicted_category_labels)

    recall = recall_score(test_labels, predicted_category_labels,average=None, zero_division = 0)

    precision = precision_score(test_labels, predicted_category_labels,average=None, zero_division = 0)

    f1 = f1_score(test_labels, predicted_category_labels,average=None, zero_division = 0)

    

    cv_accuracy_scores.append(accuracy)

    cv_recall_scores.append(recall)

    cv_precision_scores.append(precision)

    cv_f1_scores.append(f1)
print("LSTM Accuracy={}".format(np.mean(cv_accuracy_scores)))
total = 0

for x in cv_recall_scores:

    l = len(x)

    s = np.sum(x)

    total += s/l



print("LSTM Recall=" + str(total/len(cv_recall_scores)))
total = 0

for x in cv_precision_scores:

    l = len(x)

    s = np.sum(x)

    total += s/l



print("LSTM precision=" + str(total/len(cv_precision_scores)))
total = 0

for x in cv_f1_scores:

    l = len(x)

    s = np.sum(x)

    total += s/l

    

print("LSTM f1=" + str(total/len(cv_f1_scores)))
sumMat = np.zeros((len(fineLabels),len(fineLabels)))

for mat in lstm_conf_mat_fine:

    sumMat = np.add(sumMat,mat)



sumMat = pd.DataFrame(sumMat, index=fineLabels, columns=fineLabels)



print("LSTM fine class confusion matrix over 10-fold cross validation")

sumMat