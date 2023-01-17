import pandas as pd

import numpy as np



from bs4 import BeautifulSoup

import re

import nltk

# download text data sets

# nltk.download()

from nltk.corpus import stopwords

# print(stopwords.words("english"))



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv', 

                    header = 0, 

                    delimiter = '\t', 

                    quoting = 3)

# header = 0 means first line is column name

# quoting = 3 means ignore doubled quotes
print(train.shape)

print(train.head())
def review_to_words(raw_review):

    """

    convert review to a string of words, input and output are a single review and stirng of words

    """

    review_text = BeautifulSoup(raw_review).get_text()

    letters_only = re.sub("[^A-Za-z]", " ", review_text)

    words = letters_only.lower().split()

    

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    

    return(" ".join(meaningful_words))
num_reviews = train['review'].size

clean_train_reviews = []

for i in range(num_reviews):

    if (i+1) % 5000 == 0:

        print("Review %d of %d\n" % (i+1, num_reviews))

    clean_train_reviews.append(review_to_words(train["review"][i]))
vectorizer = CountVectorizer(analyzer = "word",

                             tokenizer = None,

                             preprocessor = None,

                             stop_words = None,

                             max_features = 5000)



train_data_features = vectorizer.fit_transform(clean_train_reviews)

train_data_features = train_data_features.toarray()
print(train_data_features.shape)
vocab = vectorizer.get_feature_names()

print(vocab[:10])
# return the number of appearing in each word

dist = np.sum(train_data_features, axis = 0)

for tag, count in zip(vocab[:10], dist[:10]):

    print(count, tag)
# random forest

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data_features, train['sentiment'])
# naive bayes

nb_classifier = MultinomialNB()

nb_classifier = nb_classifier.fit(train_data_features, train['sentiment'])
test = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv', header = 0, delimiter = "\t", quoting = 3)



num_reviews = len(test['review'])



clean_test_reviews = []

for i in range(num_reviews):

    clean_review = review_to_words(test['review'][i])

    clean_test_reviews.append(clean_review)

    

test_data_features = vectorizer.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()
result_forest = forest.predict(test_data_features)

result_nbc = nb_classifier.predict(test_data_features)



output_forest = pd.DataFrame(data = {"id": test["id"], "sentiment": result_forest})

output_nbc = pd.DataFrame(data = {"id": test["id"], "sentiment": result_nbc})
output_nbc.head()