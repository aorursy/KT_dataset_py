# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np 

import pandas as pd

import re

import time



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train = pd.read_fwf('/kaggle/input/pwm_seq_200bp_train_set_10k.txt', header = None)

data_train.columns = ['label', 'dna_sequence']

data_val = pd.read_fwf('/kaggle/input/pwm_seq_200bp_valid_set.txt', header = None)

data_val.columns = ['label', 'dna_sequence']

data_test = pd.read_fwf('/kaggle/input/pwm_seq_200bp_test_set_TOSEND.txt', header= None)

data_test.columns=['dna_sequence']
data_train.head(5)

data_val.head(5)
data_test.head(5)
data_train.info()
data_train.label.value_counts()
# Data Preparation 
import numpy as np

import re

def create_array_dna(dna_data):

    dna_data = dna_data.lower()

    dna_data = re.sub('[^acgt]', 'n', dna_data)

    dna_array = np.array(list(dna_data))

    return dna_array



# create a label encoder with 'acgtn' nucleotides

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

label_encoder.fit(np.array(['a','c','g','t','n']))
def ordinal_encoder(dna_data):

    original_encoding = label_encoder.transform(dna_data)

    final_encoded = original_encoding.astype(float)

    final_encoded[final_encoded == 0] = 0.25 # A

    final_encoded[final_encoded == 1] = 0.50 # C

    final_encoded[final_encoded == 2] = 0.75 # G

    final_encoded[final_encoded == 3] = 1.00 # T

    final_encoded[final_encoded == 4] = 0.00 # anything else, N

    return final_encoded
our_dna_seq = 'AACGCGCTTNN'

ordinal_encoder(create_array_dna(our_dna_seq))
from sklearn.preprocessing import OneHotEncoder

def one_hot_encoder(dna_data):

    original_encoding = label_encoder.transform(dna_data)

    one_hot_encoding = OneHotEncoder(sparse=False, dtype=int, categories=[range(5)])

    original_encoding = original_encoding.reshape(len(original_encoding), 1)

    one_hot_encoding = one_hot_encoding.fit_transform(original_encoding)

    one_hot_encoding = np.delete(one_hot_encoding, -1, 1)

    return one_hot_encoding
our_dna_seq = 'AACGCGGTTNN'

one_hot_encoder(create_array_dna(our_dna_seq))
def cal_kmer_seq(dna_data, size):

    return [dna_data[x:x+size].lower() for x in range(len(dna_data) - size + 1)]
our_dna_seq = 'ATCGATCAC'

cal_kmer_seq(our_dna_seq, size=3)
our_dna_seq = 'ATCGATCAC'

cal_kmer_seq(our_dna_seq, size=4)
our_dna_seq = 'ATCGATCAC'

cal_kmer_seq(our_dna_seq, size=5)
our_dna_seq = 'ATCGATCAC'

cal_kmer_seq(our_dna_seq, size=6)
x = cal_kmer_seq(our_dna_seq, size=6)

final_sequence = ' '.join(x)

final_sequence
def cal_kmer_seq(dna_data, size=6):

    return [dna_data[x:x+size].lower() for x in range(len(dna_data) - size + 1)]
#Our original data set preprocessing



data_train['final_sequence'] = data_train.apply(lambda x: cal_kmer_seq(x['dna_sequence']), axis=1)

data_train = data_train.drop('dna_sequence', axis=1)

data_val['final_sequence'] = data_val.apply(lambda x: cal_kmer_seq(x['dna_sequence']), axis=1)

data_val = data_val.drop('dna_sequence', axis=1)

data_test['final_sequence'] = data_test.apply(lambda x: cal_kmer_seq(x['dna_sequence']), axis=1)

data_test = data_test.drop('dna_sequence', axis=1)
data_train.head()
'''We now convert the lists of k-mers for each DNA sequence in the data into string sentences of 

    words that the count vectorizer can use for our BAG of words model. This is similar to counting the number of k-mer

    sequence as shown in the above examples.

'''



data_texts = list(data_train['final_sequence'])

for item in range(len(data_texts)):

    data_texts[item] = ' '.join(data_texts[item])

y_train = data_train.iloc[:, 0].values    



val_texts = list(data_val['final_sequence'])

for item in range(len(val_texts)):

    val_texts[item] = ' '.join(val_texts[item])

y_val = data_val.iloc[:, 0].values                      



test_texts = list(data_test['final_sequence'])

for item in range(len(test_texts)):

    test_texts[item] = ' '.join(test_texts[item])

data_texts[0]
y_train
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(4,4))

X = cv.fit_transform(data_texts)

X_val = cv.transform(val_texts)

X_test = cv.transform(test_texts)
print(X.shape)

print(X_val.shape)

print(X_test.shape)
### Multinomial Naive Bayes Classifier ###

# I have randomly chosen the alpha parameter value, we can also perform grid search CV to get the optimal value.

#This is be part of the hyper parameter tunning

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB(alpha=0.1)

classifier.fit(X, y_train)
#Checking on the validation data



y_pred = classifier.predict(X_val)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

confusion_matrix(y_val, y_pred)

print(classification_report(y_val, y_pred, labels=[0,1, 2, 3]))
#Generating our final result on test data 



y_pred_test = classifier.predict(X_test)

y_pred_test = pd.DataFrame(y_pred_test, columns=['predictions']).to_csv('prediction.csv')
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X, y_train)
#Checking on the validation data

y_pred_1 = clf.predict(X_val)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

confusion_matrix(y_val, y_pred_1)

print(classification_report(y_val, y_pred_1, labels=[0,1, 2, 3]))
#Generating our final result on test data 

rd = clf.predict(X_test)

rd = pd.DataFrame(rd, columns=['predictions']).to_csv('prediction_rd.csv')
from sklearn.svm import LinearSVC

linear = LinearSVC(random_state=0, tol=1e-5)

linear.fit(X, y_train)
#Checking on the validation data

y_pred_2 = linear.predict(X_val)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

confusion_matrix(y_val, y_pred_2)

print(classification_report(y_val, y_pred_2, labels=[0,1, 2, 3]))
#Generating our final result on test data 

linear_res = linear.predict(X_test)

linear_res = pd.DataFrame(linear_res, columns=['predictions']).to_csv('prediction_linear.csv')