import os

print(os.listdir("../input"))
import re    # for regular expressions 

import nltk  # for text manipulation 

import string 

import warnings 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt  

pd.set_option("display.max_colwidth", 200) 

warnings.filterwarnings("ignore", category=DeprecationWarning) 

train  = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/train.csv') 

test = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/test.csv')



train[train['label'] == 1].head(10)
length_train = train['tweet'].str.len() 

length_test = test['tweet'].str.len() 

plt.hist(length_train, bins=20, label="train_tweets") 

plt.hist(length_test, bins=20, label="test_tweets") 

plt.legend() 

plt.show()
combi = train.append(test, ignore_index=True) 

combi.shape

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

    return input_txt   



combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") 

combi.head()



combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") 

combi.head(10)



combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

combi.head(10)



tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing 

tokenized_tweet.head()



from nltk.stem.porter import * 

stemmer = PorterStemmer() 

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

tokenized_tweet.head()



for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])    

combi['tidy_tweet'] = tokenized_tweet



combi.head(10)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import gensim
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

bow.shape
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing 

model_w2v = gensim.models.Word2Vec(

            tokenized_tweet,

            size=200, # desired no. of features/independent variables

            window=5, # context window size

            min_count=2,

            sg = 1, # 1 for skip-gram model

            hs = 0,

            negative = 10, # for negative sampling

            workers= 2, # no.of cores

            seed = 34) 
model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)
model_w2v.wv.most_similar(positive="dinner")
model_w2v.wv.most_similar(positive="trump")
model_w2v['india']
from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split 

from sklearn.metrics import f1_score
# Extracting train and test BoW features 

train_bow = bow[:31962,:] 

test_bow = bow[31962:,:] 

# splitting data into training and validation set 

xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],random_state=42,                                                           test_size=0.3)
lreg = LogisticRegression() 

# training the model 

lreg.fit(xtrain_bow, ytrain) 

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set 

prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0 prediction_int = prediction_int.astype(np.int) 

f1_score(yvalid, prediction_int) # calculating f1 score for the validation set