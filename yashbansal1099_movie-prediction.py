import pandas as pd

import numpy as np

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer





def stemmed(review):

    review = review.lower()

    review = review.replace("<br /><br />", " ")

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





x = pd.read_csv("../input/movie-rating/Train.csv").values

y = x[:, 1]

x = x[:, 0]



for i in range (len(y)):

    if (y[i] == 'pos'):

        y[i] = 1

    else:

        y[i] = 0

y = list(y)

tokenizer = RegexpTokenizer(r'\w+')

en_stopwords = set(stopwords.words('english'))

ps = PorterStemmer()

cleaned = stemdoc(x)



from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer()

x_voc = cv.fit_transform(cleaned)

x_t = pd.read_csv("../input/movie-rating/Test.csv").values

x_t = x_t[:, 0]

x_t_cleaned = stemdoc(x_t)

xt_voc = cv.transform(x_t_cleaned)

print(xt_voc.shape)





from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB



mnb = MultinomialNB()

mnb.fit(x_voc, y)



yt = mnb.predict(xt_voc)



print(yt)

yo = []

for i in range(len(yt)):

    if (yt[i]==1):

        yo.append('pos')

    else:

        yo.append('neg')

n = np.arange(10000)



print(yo)

yt = np.asarray(yo)

y_out = np.stack((n, yt), axis = 1)

df = pd.DataFrame(y_out, columns = ["Id","label"])

print(df)

df.to_csv('OUTPUT.csv', index = False)