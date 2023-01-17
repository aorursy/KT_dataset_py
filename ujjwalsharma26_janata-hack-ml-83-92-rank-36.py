# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv')

test = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')

samp = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv')
train.head()



l = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']



for col in l:

    print(col,':\n',train[col].value_counts())
dic = {'CS' :8594,'Phy' :6013,'Math' :5618,'Stats' :5206,'QB' :587,'QF' :249}



values = dic.values()

total = sum(values)

percent_values = [value * 100. / total for value in values]

print(percent_values)
test = test.drop(['ID'],axis=1)
X = train.loc[:,['TITLE','ABSTRACT']]

y = train.loc[:,l]



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.16, random_state=42)
y_test.reset_index(drop=True,inplace=True)

X_test.reset_index(drop=True,inplace=True)
y1 = np.array(y_train)

y2 = np.array(y_test)


X_train.replace('[^a-zA-Z]',' ', regex=True, inplace=True)

X_test.replace('[^a-zA-Z]',' ', regex=True, inplace=True)



test.replace('[^a-zA-Z]',' ', regex=True, inplace=True)
for index in X_train.columns:

    X_train[index] = X_train[index].str.lower()



for index in X_test.columns:

    X_test[index] = X_test[index].str.lower()



for index in test.columns:

    test[index] = test[index].str.lower()
X_train['ABSTRACT'] = X_train['ABSTRACT'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')

X_test['ABSTRACT'] = X_test['ABSTRACT'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')



test['ABSTRACT'] = test['ABSTRACT'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
X_train = X_train.replace('\s+', ' ', regex=True)

X_test = X_test.replace('\s+', ' ', regex=True)



test = test.replace('\s+', ' ', regex=True)
import nltk

nltk.download('punkt')

nltk.download('wordnet')

nltk.download('stopwords')

nltk.download('averaged_perceptron_tagger')

from nltk import sent_tokenize, word_tokenize

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 

len(stop_words)

X_train['ABSTRACT'] = X_train['ABSTRACT'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

X_test['ABSTRACT'] = X_test['ABSTRACT'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))



test['ABSTRACT'] = test['ABSTRACT'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
X_train['combined'] = X_train['TITLE']+' '+X_train['ABSTRACT']

X_test['combined'] = X_test['TITLE']+' '+X_test['ABSTRACT']



test['combined'] = test['TITLE']+' '+test['ABSTRACT']
X_train = X_train.drop(['TITLE','ABSTRACT'],axis=1)

X_test = X_test.drop(['TITLE','ABSTRACT'],axis=1)



test = test.drop(['TITLE','ABSTRACT'],axis=1)
X_train.head()
train_lines = []

for row in range(0,X_train.shape[0]):

    train_lines.append(' '.join(str(x) for x in X_train.iloc[row,:]))



test_lines = []

for row in range(0,X_test.shape[0]):

    test_lines.append(' '.join(str(x) for x in X_test.iloc[row,:]))



predtest_lines = []

for row in range(0,test.shape[0]):

    predtest_lines.append(' '.join(str(x) for x in test.iloc[row,:]))
X.replace('[^a-zA-Z]',' ', regex=True, inplace=True)

for index in X.columns:

    X[index] = X[index].str.lower()

X['ABSTRACT'] = X['ABSTRACT'].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')

X = X.replace('\s+', ' ', regex=True)

X['ABSTRACT'] = X['ABSTRACT'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

X['combined'] = X['TITLE']+' '+X['ABSTRACT']
X = X.drop(['TITLE','ABSTRACT'],axis=1)
X_lines = []

for row in range(0,X.shape[0]):

    X_lines.append(' '.join(str(x) for x in X.iloc[row,:]))
from sklearn.feature_extraction.text import CountVectorizer



countvector = CountVectorizer(ngram_range=(4,8),analyzer='char',lowercase=False,strip_accents='unicode')

X_train_cv = countvector.fit_transform(train_lines)

X_test_cv = countvector.transform(test_lines)



test_cv = countvector.transform(predtest_lines)
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer



tfidfvector = TfidfTransformer(sublinear_tf=True,use_idf=True)

# smooth_idf=False

X_train_tf = tfidfvector.fit_transform(X_train_cv)

X_test_tf = tfidfvector.fit_transform(X_test_cv)



test_tf = tfidfvector.fit_transform(test_cv)
X_cv = countvector.transform(X_lines)



X_tf = tfidfvector.fit_transform(X_cv)
from sklearn.svm import LinearSVC

from sklearn.multioutput import MultiOutputClassifier

from sklearn.multiclass import OneVsRestClassifier



model = LinearSVC(class_weight='balanced',loss="hinge",fit_intercept=False)

models = MultiOutputClassifier(model)



# text_clf = Pipeline([('tfidf', TfidfVectorizer(min_df=True,smooth_idf=True,sublinear_tf=True,analyzer='char',strip_accents='ascii',token_pattern=r'(?ui)\\b\\w*[a-z]+\\w*\\b')),

#                          ('clf',LinearSVC(loss="hinge",intercept_scaling=1.05 ,class_weight='balanced')),

#     ])
models.fit(X_tf, y)
preds = models.predict(X_test_tf)

preds
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



print(classification_report(y2,preds))

print(accuracy_score(y2,preds))
predssv = models.predict(test_tf)

predssv
test1 = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')

submit = pd.DataFrame({'ID': test1.ID, 'Computer Science': predssv[:,0],'Physics':predssv[:,1],'Mathematics':predssv[:,2],'Statistics':predssv[:,3],'Quantitative Biology':predssv[:,4],'Quantitative Finance':predssv[:,5]})

submit.head()
submit.to_csv('submission.csv', index=False)