import os
import gzip
import pandas as pd
X=list()
for dirname,_,filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path=os.path.join(dirname,filename)
        if path.endswith('Dataset.csv'):
            X.append(path)
'''
with gzip.open(X[3],'r') as f:
    data=f.read()
data=str(data)
'''
X=pd.read_csv(X[0])
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer,PorterStemmer
lemmatizer=WordNetLemmatizer()
tokenizer=nltk.tokenize.TreebankWordTokenizer()
#stemmer=PorterStemmer()
def convert(review):
    tokens=tokenizer.tokenize(review)
    #import spacy
    #nlp=spacy.load('en',disable=['parser','ner'])
    #XY=nlp(' '.join(tokens))
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    Y=[lemmatizer.lemmatize(token,get_wordnet_pos(token)) for token in tokens]
    return ' '.join(Y)
X_train=pd.DataFrame(index=range(len(X['review'])),columns=['review','sentiment'])
for _ in range(len(X['review'])):
    if _%100==0:
        print(_)
    X_train['review'][_]=convert(X['review'][_])
    X_train['sentiment'][_]=X['sentiment'][_]
X_train=X_train.sample(frac=1)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_df=0.7,strip_accents='ascii',ngram_range=(1,2))
features=vectorizer.fit_transform(X_train['review'])
X_features=pd.DataFrame(features.todense(),columns=vectorizer.get_feature_names())
X_features
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.optimizers import Adam
inp_size=len(X_features.columns)
model=Sequential()
model.add(Input(shape=(inp_size,)))
model.add(Dense(1,activation='sigmoid'))
optimizer=Adam(lr=0.001)
model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['binary_accuracy'])
import numpy as np
x=np.array(features.todense())
y=np.array([1 if X_train['sentiment'][_]=='positive' else 0 for _ in range(len(X['review']))])
model.fit(x,y,validation_split=0.1,verbose=2,epochs=10000,shuffle=True,batch_size=256)
!pip install google.colab
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    ! wget https://raw.githubusercontent.com/hse-aml/natural-language-processing/master/setup_google_colab.py -O setup_google_colab.py
    import setup_google_colab
    setup_google_colab.setup_week1() 
    
import sys
sys.path.append("..")
from common.download_utils import download_week1_resources

download_week1_resources()
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from ast import literal_eval
import numpy as np
import pandas as pd
def read_data(filename):
    X=pd.read_csv(filename,sep='\t')
    #X['tags']=X['tags'].apply(lambda s: '\"'+s+'\"')
    X['tags']=X['tags'].apply(literal_eval)
    return X

tr_data=read_data('data/train.tsv')
val_data=read_data('data/validation.tsv')
te_data=pd.read_csv('data/test.tsv',sep='\t')

X_train,y_train=tr_data['title'].values,tr_data['tags'].values
X_val,y_val=val_data['title'].values,val_data['tags'].values
X_test=te_data['title'].values
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ',text)
    text = BAD_SYMBOLS_RE.sub('',text)
    text = ' '.join([_ if _ not in STOPWORDS else '' for _ in text.split(' ')])
    return re.sub('\s+',' ',text).strip()

def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'
print(test_text_prepare())
from grader import Grader
grader=Grader()
prepared_questions = []
for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)

grader.submit_tag('TextPrepare', text_prepare_results)
X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]
from collections import Counter
words=list()
tags=list()
[[words.append(b) for b in _.split(' ')] for _ in X_train]
[[tags.append(t) for t in _] for _ in y_train]
most_common_words=Counter(words).most_common(3)
most_common_tags=Counter(tags).most_common(3)
grader.submit_tag('WordsTagsCount', '%s\n%s' % (','.join(tag for tag, _ in most_common_tags),','.join(word for word, _ in most_common_words)))
import numpy as np
DICT_SIZE = 5000
WORDS_TO_INDEX=dict()
INDEX_TO_WORDS=dict()
W=Counter(words).most_common(DICT_SIZE)
for w,idx in zip(W,range(0,DICT_SIZE)):
    WORDS_TO_INDEX[w[0]]=idx
    INDEX_TO_WORDS[idx]=w[0]
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    result_vector = np.zeros(dict_size)
    for _ in text.split(' '):
        if _ in words_to_index.keys():
            result_vector[words_to_index[_]]=1
    return result_vector

def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

print(test_my_bag_of_words())
from scipy import sparse as sp_sparse
X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)
row = X_train_mybag[10].toarray()[0]
non_zero_elements_count = np.count_nonzero(row)
grader.submit_tag('BagOfWords', str(non_zero_elements_count))
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
def tfidf_features(X_train, X_val, X_test):
    tfidf_vectorizer=TfidfVectorizer(min_df=5,max_df=0.9,token_pattern='(\S+)',ngram_range=(1,2))
    X_train=tfidf_vectorizer.fit_transform(X_train)
    X_val=tfidf_vectorizer.transform(X_val)
    X_test=tfidf_vectorizer.transform(X_test)
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_
X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}
tags_count=Counter(tags)
from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer(classes=sorted(tags_count.keys()))
y_train=mlb.fit_transform(y_train)
y_val=mlb.fit_transform(y_val)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
def train_classifier(X_train,y_train):
    ovrc=OneVsRestClassifier(LogisticRegression(solver='saga',penalty='elasticnet',l1_ratio=0.5,max_iter=1000,multi_class='multinomial',verbose=2,n_jobs=-1),n_jobs=-1)
    #ovrc=OneVsRestClassifier(LinearSVC(),n_jobs=-1)
    #ovrc=OneVsRestClassifier(MultinomialNB(fit_prior=True,class_prior=None),n_jobs=-1)
    #ovrc=OneVsRestClassifier(RidgeClassifier(max_iter=1000),n_jobs=-1)
    ovrc=ovrc.fit(X_train,y_train)
    return ovrc
classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)
y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
#y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
#y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
#from sklearn.metrics import recall_score
def print_evaluation_scores(y_val, predicted):
    print('Accuracy Score: {} F1 Score: {} Precision Score: {}'.format(accuracy_score(y_val,predicted),f1_score(y_val,predicted,average='macro'),average_precision_score(y_val,predicted,average='macro')))
print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)
test_predictions = classifier_mybag.predict(X_test_mybag)
test_pred_inversed = mlb.inverse_transform(test_predictions)

test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
grader.submit_tag('MultilabelClassification', test_predictions_for_submission)