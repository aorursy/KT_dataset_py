import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
sample=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
train.head()
train.info()
train.describe()
train.shape
BASE = "/kaggle/input/nlp-getting-started/"

train = pd.read_csv(BASE + "train.csv")

test = pd.read_csv(BASE + "test.csv")

sub = pd.read_csv(BASE + "sample_submission.csv")
tweets = train[['text', 'target']]

tweets.head()
tweets.target.value_counts()
tweets.shape
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer
def remove_punctuation(text):

    '''a function for removing punctuation'''

    import string

    # replacing the punctuations with no space, 

    # which in effect deletes the punctuation marks 

    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks

    return text.translate(translator)
tweets['text'] = tweets['text'].apply(remove_punctuation)

tweets.head(10)
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
tweets['text'] = tweets['text'].apply(stopwords)

tweets.head(10)
# create an object of stemming function

stemmer = SnowballStemmer("english")



def stemming(text):    

    '''a function which stems each word in the given text'''

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text)
tweets['text'] = tweets['text'].apply(stemming)

tweets.head(10)
vectorizer = CountVectorizer(analyzer='word', binary=True)

vectorizer.fit(tweets['text'])
X = vectorizer.transform(tweets['text']).todense()

y = tweets['target'].values

X.shape, y.shape
from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split



from sklearn.metrics import f1_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)



f1score = f1_score(y_test, y_pred)

print(f"Model Score: {f1score * 100} %")
tweets_test = test['text']

test_X = vectorizer.transform(tweets_test).todense()

test_X.shape
lr_pred = model.predict(test_X)
sub['target'] = lr_pred

sub.to_csv("submission.csv", index=False)

sub.head()