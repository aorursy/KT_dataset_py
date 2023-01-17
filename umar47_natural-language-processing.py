import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import OneHotEncoder
import os
import regex as re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.backend import eval
from keras.optimizers import Adam
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D,MaxPooling1D
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/nyc-jobs.csv')
pd.set_option('display.max_columns', None)
df=df.drop(['Job ID', 'Agency', 'Posting Type', '# Of Positions', 'Business Title',
       'Civil Service Title', 'Title Code No', 'Level', 'Job Category',
       'Full-Time/Part-Time indicator' , 'Additional Information', 'To Apply', 'Hours/Shift', 'Work Location 1',
       'Recruitment Contact', 'Residency Requirement', 'Posting Date',
       'Post Until', 'Posting Updated', 'Process Date'], axis=1)
df=df.reset_index(drop=True)
print(df.columns, df.tail(7))

print(df.info())
#df['Preferred Skills'].dropna(inplace=True)
#print(len(df['Preferred Skills']))
#df[['Preferred Skills'].fillna('Unspecified', inplace=True)
X=df['Job Description']
ohe=OneHotEncoder()
y=df[['Salary Range From']].astype('str')
print(y.info())
import string
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
def object_to_list(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]#removing puctuations
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]#removing non english words

from sklearn.feature_extraction.text import CountVectorizer
# next we need to vectorize our input variable (X)
#we use the count vectoriser function and the analyser we use is the above lines of code
# this should return a vector array
X = CountVectorizer(analyzer=object_to_list).fit_transform(X)
print(X[6].split())
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
#ohe=OneHotEncoder()
#ohe.fit_transform(X)
x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble  import RandomForestClassifier
first_model=RandomForestClassifier()
first_model.fit(x_train, y_train)
print(first_model.score(x_train, y_train))

from sklearn.metrics import confusion_matrix, classification_report
predicted=first_model.predict(x_test)
print(confusion_matrix(y_test, predicted))
print('\n')
print(classification_report(y_test, predicted))#we see the precision, recall, f1-score and supprt for 
# predicted values here
def clean_document(doco):
    punctuation = string.punctuation
    punc_replace = ''.join([' ' for s in punctuation])
    doco_link_clean = re.sub(r'http\S+', '', doco)
    doco_clean_and = re.sub(r'&\S+', '', doco_link_clean)
    doco_clean_at = re.sub(r'@\S+', '', doco_clean_and)
    doco_clean = doco_clean_at.replace('-', ' ')
    doco_alphas = re.sub(r'\W +', ' ', doco_clean)
    trans_table = str.maketrans(punctuation, punc_replace)
    doco_clean = ' '.join([word.translate(trans_table) for word in doco_alphas.split(' ')])
    doco_clean = doco_clean.split(' ')
    p = re.compile(r'\s*\b(?=[a-z\d]*([a-z\d])\1{3}|\d+\b)[a-z\d]+', re.IGNORECASE)
    doco_clean = ([p.sub("", x).strip() for x in doco_clean])
    doco_clean = [word.lower() for word in doco_clean if len(word) > 2]
    doco_clean = ([i for i in doco_clean if i not in stop])
#     doco_clean = [spell(word) for word in doco_clean]
#     p = re.compile(r'\s*\b(?=[a-z\d]*([a-z\d])\1{3}|\d+\b)[a-z\d]+', re.IGNORECASE)
    doco_clean = ([p.sub("", x).strip() for x in doco_clean])
#     doco_clean = ([spell(k) for k in doco_clean])
    return doco_clean
reviews=df['Job Description']
review_cleans = [clean_document(doc) for doc in reviews];
sentences = [' '.join(r) for r in review_cleans ]
print(reviews.shape)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
text_sequences = np.array(tokenizer.texts_to_sequences(sentences))
sequence_dict = tokenizer.word_index
word_dict = dict((num, val) for (val, num) in sequence_dict.items())
print(text_sequences)
#print(sequence_dict)

