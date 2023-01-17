import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output



%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/train.csv").fillna("")

df.head() 
df_test = pd.read_csv("../input/test.csv").fillna("")

df.info()

df_test.info()
data=df.head(50000)

data2=df_test.head(10000)
import pickle

import pandas as pd

import numpy as np

import gensim

from fuzzywuzzy import fuzz

from nltk.corpus import stopwords

from tqdm import tqdm

from scipy.stats import skew, kurtosis

from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

from nltk import word_tokenize

stop_words = stopwords.words('english')
from nltk.corpus import stopwords



stops = set(stopwords.words("english"))



def word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    return R





train_word_match = data.apply(word_match_share, axis=1, raw=True)
from sklearn.metrics import roc_auc_score

print('Original AUC:', roc_auc_score(data['is_duplicate'], train_word_match))
data.info()

data2.info()
data = data.drop(['id', 'qid1', 'qid2'], axis=1)

data2 = data2.drop(['test_id'], axis=1)
print("start generating fs1...")

data['len_q1'] = data.question1.apply(lambda x: len(str(x))) 

data['len_q2'] = data.question2.apply(lambda x: len(str(x))) 

data['diff_len'] = data.len_q1 - data.len_q2 

data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', ''))))) 

data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', ''))))) 

data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split())) 

data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split())) 

data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)

#data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

#data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

#data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data2['len_q1'] = data2.question1.apply(lambda x: len(str(x))) 

data2['len_q2'] = data2.question2.apply(lambda x: len(str(x))) 

data2['diff_len'] = data2.len_q1 - data2.len_q2 

data2['len_char_q1'] = data2.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', ''))))) 

data2['len_char_q2'] = data2.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', ''))))) 

data2['len_word_q1'] = data2.question1.apply(lambda x: len(str(x).split())) 

data2['len_word_q2'] = data2.question2.apply(lambda x: len(str(x).split())) 

data2['common_words'] = data2.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
data2['fuzz_qratio'] = data2.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data2['fuzz_WRatio'] = data2.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
data2['fuzz_partial_ratio'] = data2.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)

#data2['fuzz_partial_token_set_ratio'] = data2.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

#data2['fuzz_partial_token_sort_ratio'] = data2.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

#data2['fuzz_token_set_ratio'] = data2.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data2['fuzz_token_sort_ratio'] = data2.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data.info()

data2.info()
x_train = pd.DataFrame()

x_test = pd.DataFrame()

x_train=data.iloc[:,3:]

x_train['word_match'] = train_word_match

x_test=data2.iloc[:,2:]

x_train.info()

x_test.info()
x_train = x_train.drop(['len_q1', 'len_q2', 'len_char_q1','len_char_q2','len_word_q1','len_word_q2'], axis=1)

x_test = x_test.drop(['len_q1', 'len_q2', 'len_char_q1','len_char_q2','len_word_q1','len_word_q2'], axis=1)

x_train.info()
test_word_match = data2.apply(word_match_share, axis=1, raw=True)

x_test['word_match'] = test_word_match

x_test.info()
x_test.head(15)
y_train = data['is_duplicate'].values
pos_train = x_train[y_train == 1]

neg_train = x_train[y_train == 0]



# Now we oversample the negative class

# There is likely a much more elegant way to do this...

p = 0.165

scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

while scale > 1:

    neg_train = pd.concat([neg_train, neg_train])

    scale -=1

neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

print(len(pos_train) / (len(pos_train) + len(neg_train)))



x_train = pd.concat([pos_train, neg_train])

y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

del pos_train, neg_train
from sklearn.cross_validation import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'reg:logistic'

params['eval_metric'] = 'logloss'

params['subsample']=1.0

params['min_child_weight']=5

params['colsample_bytree']=0.2

params['eta'] = 0.1

params['max_depth'] = 8



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=10)