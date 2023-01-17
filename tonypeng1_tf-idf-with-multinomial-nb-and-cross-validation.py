import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import string

from time import time

%matplotlib inline
from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords
# import nltk

# nltk.download()

# (The above is the code that may be needed the first time nltk libraries are downloaded.)
data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')

data.head()
data.info()
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

data.head()
data.columns = ['class', 'text']

data.head()
count_class = pd.value_counts(data["class"])

count_class
for i in range(5): 

    print(data['text'][i])
stopwords_list = stopwords.words('english')

stopwords_list
data['text'][4]
words_new = [i for i in data['text'][4].split() if i.lower() not in stopwords.words('english')]

' '.join(words_new)
punc_string = string.punctuation

print(punc_string)

print(len(punc_string))
data['text'][2]
data['text'][2].translate(str.maketrans(' ', ' ', punc_string))  
data['text'][2].translate(str.maketrans(punc_string, len(punc_string)*' '))
print(data['text'][3])

print(data['text'][4])
stemmer = SnowballStemmer('english')

print(' '.join([stemmer.stem(word) for word in data['text'][3].split()]))

print(' '.join([stemmer.stem(word) for word in data['text'][4].split()]))
def preprocess(text):

    # Remove stopwords

    words = [i for i in text.split() if i.lower() not in stopwords_list]

    text_1 = ' '.join(words)

    # Remove puctuation and replace with a space.

    text_2 = text_1.translate(str.maketrans(punc_string, len(punc_string)*' '))

    words_1 = text_2.split()

    # Perform word stemming 

    words_2 = [stemmer.stem(word) for word in words_1]

    text_3 = ' '.join(words_2)

    return text_3
print(data['text'][4])

preprocess(data['text'][4])
data['text'] = data['text'].apply(preprocess)
data.head(10)
count_v = CountVectorizer()

word_count_matrix = count_v.fit_transform(data['text'])

word_count_matrix
print(word_count_matrix)
count_list = word_count_matrix.toarray().sum(axis=0)

word_list = count_v.get_feature_names()
word_freq = pd.DataFrame(count_list, index=word_list, columns=['Freq'])

word_freq.sort_values(by='Freq', ascending=False).head(30)
data['text'][23]
text_freq = pd.DataFrame(word_count_matrix.toarray()[23], index=count_v.get_feature_names(), columns=['word freq'])

# remove the rows with 0 frequency count

text_freq = text_freq[text_freq['word freq']!=0]

text_freq
tf_idf = TfidfTransformer()

tf_idf.fit(word_count_matrix)

tf_idf.idf_
tf_idf.idf_.shape
idf = pd.DataFrame(tf_idf.idf_, index=count_v.get_feature_names(), columns=['idf_weight'])

idf.sort_values(by='idf_weight')
print('max = ' + str(tf_idf.idf_.max()))

print('min = ' + str(tf_idf.idf_.min()))

print('mean = ' + str(tf_idf.idf_.mean()))
word_list_to_drop =[i for i in idf.index if i not in text_freq.index]

# (the list of words that need to be dropped)

idf_1 = idf.drop(word_list_to_drop)

idf_1
tf_idf_vector = tf_idf.transform(word_count_matrix)

tf_idf_vector
tf_idf_1 = pd.DataFrame(tf_idf_vector.toarray()[23], index=count_v.get_feature_names(), columns=['tf_idf'])

# remove the rows with tf-idf = 0

tf_idf_1 = tf_idf_1[tf_idf_1['tf_idf']!=0.0]

tf_idf_1
df = pd.concat([text_freq, idf_1, tf_idf_1], axis=1)

df
df['(word freq)x(idf_weight)'] = df['word freq'] * df['idf_weight']

df = df.reindex(['word freq', 'idf_weight', '(word freq)x(idf_weight)', 'tf_idf'], axis=1)

df
df_1 = pd.DataFrame(df['(word freq)x(idf_weight)']).T

# (Perform a transpose of the "(word freq)x(idf_weight)" column)

df_1
# Perform l2 normalization on the "(word freq)x(idf_weight)" vector using the "normalize" function.

normalize(df_1)
np.sum(np.square(normalize(df_1)))
data.head()
X = data['text']

y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = Pipeline([

    ('vector', CountVectorizer()), 

    ('tfidf', TfidfTransformer()), 

    ('mulNB', MultinomialNB())

])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
parameters = {

    'mulNB__alpha': [1, 0.7, 0.4, 0.2, 0.1, 0.09, 0.08, 0.07, 0.06, 0.03, 0.01] 

}



grid = GridSearchCV(pipe, param_grid=parameters, cv=10, refit=True)

t0 = time()

grid.fit(X_train, y_train)

print("done in %0.3fs" % (time() - t0))
print(grid.best_score_)

print(grid.best_params_)
grid.score(X_test, y_test)