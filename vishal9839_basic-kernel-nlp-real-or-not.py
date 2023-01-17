# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        import re # for regular expressions

import pandas as pd 

pd.set_option("display.max_colwidth", 200)

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

import string

import nltk # for text manipulation

import warnings

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train.head()
train.target.value_counts()
train.shape
train = train.drop(['keyword'], axis=1)

train = train.drop(['location'], axis=1)
##Remove Pattern with username @

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

        

    return input_txt
train['text'] = np.vectorize(remove_pattern)(train['text'], "@[\w]*") 

train
##Remove all the hyperlinks from the texts

train['text'] = train['text'].str.replace('http\S+|www.\S+', '', case=False)



train['text'] = train['text'].str.replace("[^a-zA-Z#]", " ")
##Removing all the words with less than 3 characters

train['text'] = train['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
##Removing stopwords

# extracting the stopwords from nltk library

sw = stopwords.words('english')

# displaying the stopwords

np.array(sw);
def stopwords(text):

    '''a function for removing the stopword'''

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # joining the list of words with space separator

    return " ".join(text)
train['text'] = train['text'].apply(stopwords)

train.head(10)
# function to collect hashtags

def hashtag_extract(x):

    hashtags = []

    # Loop over the words in the tweet

    for i in x:

        ht = re.findall(r"#(\w+)", i)

        hashtags.append(ht)



    return hashtags
HT_disaster = hashtag_extract(train['text'][train['target'] == 1])



# extracting hashtags from racist/sexist tweets

HT_no_disaster = hashtag_extract(train['text'][train['target'] == 0])



# unnesting list

HT_disaster = sum(HT_disaster,[])

HT_no_disaster = sum(HT_no_disaster,[])
a = nltk.FreqDist(HT_disaster)

d = pd.DataFrame({'Hashtag': list(a.keys()),

                  'Count': list(a.values())})



# selecting top 20 most frequent hashtags     

d = d.nlargest(columns="Count", n = 20) 

plt.figure(figsize=(16,5))

ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

ax.set(ylabel = 'Count')

plt.show()
b = nltk.FreqDist(HT_no_disaster)

e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})



# selecting top 20 most frequent hashtags

e = e.nlargest(columns="Count", n = 20)   

plt.figure(figsize=(16,5))

ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
##Stemming Operation

# create an object of stemming function

stemmer = SnowballStemmer("english")



def stemming(text):    

    '''a function which stems each word in the given text'''

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text) 
train['text'] = train['text'].apply(stemming)
##Vectorizing the Text

vectorizer = CountVectorizer(analyzer='word', binary=True)

vectorizer.fit(train['text'])
X = vectorizer.transform(train['text']).todense()

y = train['target'].values

X.shape, y.shape
##Machine learning the model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
##Splitting the data into test and train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
model = LogisticRegression()

model.fit(X_train, y_train)
##Evaluate the Model

prediction_log = model.predict(X_test) # predicting on the validation set



f1score = f1_score(y_test, prediction_log)

print(f"Model Score: {f1score * 100} %")
sub_test = test['text']

test_X = vectorizer.transform(sub_test).todense()

test_X.shape
pred_test = model.predict(test_X)
sub['target'] = pred_test

sub.to_csv("submission.csv", index=False)

sub.head()