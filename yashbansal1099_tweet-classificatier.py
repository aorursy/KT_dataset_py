import pandas as pd

import numpy as np

from nltk.corpus import stopwords

from nltk.stem.lancaster import LancasterStemmer

from nltk.tokenize import WhitespaceTokenizer



def stemmed(review):

    review = review.lower()

    review = review.replace("\n", " ")



    tokens = tokenizer.tokenize(review)

    new_token = [token for token in tokens if token not in en_stopwords]

    stemmed_tokens = [ps.stem(token) for token in new_token]

    cleaned_review = ' '.join(stemmed_tokens)

    return cleaned_review





def stemdoc(files):

    cl = []

    for i in range(len(files)):

        cl.append(stemmed(files[i]))

    return cl
x = pd.read_csv("../input/tweet-classification/train_data.csv").values

y = list(x[:, 1])

x = x[:, 2]
print(x)
tokenizer = WhitespaceTokenizer()

en_stopwords = set(stopwords.words('english'))

ps = LancasterStemmer()

cleaned = stemdoc(x)

cleaned = np.array(cleaned)
print(cleaned)
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()

x_voc = cv.fit_transform(cleaned)

x_t = pd.read_csv("../input/tweet-classification/test_data.csv").values

index = x_t[:,0]

x_t = x_t[:, 1]

x_t_cleaned = stemdoc(x_t)

xt_voc = cv.transform(x_t_cleaned)

print(xt_voc.shape)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

from sklearn.linear_model import SGDClassifier



mnb = SGDClassifier(loss = 'modified_huber', shuffle = False, n_jobs = -1, learning_rate = 'invscaling', eta0 = 0.5)

mnb.fit(x_voc, y)
yt = mnb.predict(xt_voc)
print(yt)
yt = np.array(yt)
out = list(zip(index, yt))

df = pd.DataFrame(out, columns = ["index", "User"])

print(df)

df.to_csv('output.csv', index = False)