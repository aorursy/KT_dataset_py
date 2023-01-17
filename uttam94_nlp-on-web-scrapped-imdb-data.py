from bs4 import BeautifulSoup
! ls ../input/imdb-reviews
soup = BeautifulSoup(open("../input/imdb-reviews/1.html",encoding="utf8"), "html.parser")
movie_containers = soup.find_all('div' , class_ = 'review-container')

print(type(movie_containers))

print(len(movie_containers))
first_movie = movie_containers[0]

first_movie.a.text
temp = first_movie.span.text
temp
# Lists to store the scraped data in

reviews = []

ratings = []



# Extract data from individual movie container

for container in movie_containers:

    

    #review

    review = container.a.text

    reviews.append(review)

    

    #rating

    rating = container.span.text

    ratings.append(rating)

   
import pandas as pd

import numpy as np



test_df = pd.DataFrame({'Review': reviews,'Rating': ratings})

print(test_df.info())

test_df.head()
test_df.loc[:, 'Rating'] = test_df['Rating'].str[6:8]
test_df.loc[:, 'Rating'] = test_df['Rating'].str.replace('/', '')

test_df.loc[:, 'Review'] = test_df['Review'].str.replace('\n', '')

test_df.loc[:, 'Rating'] = test_df['Rating'].str.replace('-', '')
import re

def split_it(rating):

    return re.sub('[a-zA-Z]+','NaN', rating)
test_df['Rating'] = test_df['Rating'].apply(split_it)
test_df = test_df[test_df.Rating.str.contains("NaN") == False]
test_df['Rating'] = test_df['Rating'].apply(pd.to_numeric)
test_df.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.countplot(test_df['Rating'])
sns.countplot(test_df['Rating'])
test_df.describe()
test_df['Review']=test_df['Review'].astype(str)

test_df['Review Length']=test_df['Review'].apply(len)



g = sns.FacetGrid(data=test_df, col='Rating')

g.map(plt.hist, 'Review Length', bins=50)
plt.figure(figsize=(10,10))

sns.boxplot(x='Rating', y='Review Length', data=test_df)
from collections import Counter

from nltk.tokenize import RegexpTokenizer

from stop_words import get_stop_words

import nltk

from nltk.corpus import stopwords

from nltk import sent_tokenize, word_tokenize
nltk.download('punkt')
a = test_df['Review'].str.lower().str.cat(sep= ' ')
# removes punctuation,numbers and returns list of words

b = re.sub('[^A-Za-z]+', ' ', a)
#remove all the stopwords from the text

stop_words = list(get_stop_words('en'))         

nltk_words = list(stopwords.words('english'))   

stop_words.extend(nltk_words)



newStopWords = ['game','thrones', 'bran', 'stark', 'dragons']

stop_words.extend(newStopWords)
word_tokens = word_tokenize(b)
len(word_tokens)
filtered_sentence = []

for w in word_tokens:

    if w not in stop_words:

        filtered_sentence.append(w)
len(filtered_sentence)
# Remove characters which have length less than 2  

without_single_chr = [word for word in filtered_sentence if len(word) > 2]



# Remove numbers

cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]   
top_N = 100

word_dist = nltk.FreqDist(cleaned_data_title)

rslt = pd.DataFrame(word_dist.most_common(top_N),

                    columns=['Word', 'Frequency'])



plt.figure(figsize=(10,10))

sns.set_style("whitegrid")

ax = sns.barplot(x="Word",y="Frequency", data=rslt.head(7))
from wordcloud import WordCloud, STOPWORDS
def wc(data,bgcolor,title):

    plt.figure(figsize = (100,100))

    wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)

    wc.generate(' '.join(data))

    plt.imshow(wc)

    plt.axis('off')
wc(cleaned_data_title,'black','Most Used Words')
from textblob import TextBlob



bloblist_desc = list()



df_review_str=test_df['Review'].astype(str)
for row in df_review_str:

    blob = TextBlob(row)

    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))

    df_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['Review','sentiment','polarity'])
df_polarity_desc.head()
def f(df_polarity_desc):

    if df_polarity_desc['sentiment'] >= 0:

        val = "Positive Review"

    elif df_polarity_desc['sentiment'] >= -0.09:

        val = "Neutral Review"

    else:

        val = "Negative Review"

    return val
df_polarity_desc['Sentiment_Type'] = df_polarity_desc.apply(f, axis=1)



plt.figure(figsize=(10,10))

sns.set_style("whitegrid")

ax = sns.countplot(x="Sentiment_Type", data=df_polarity_desc)
positive_reviews=df_polarity_desc[df_polarity_desc['Sentiment_Type']=='Positive Review']

negative_reviews=df_polarity_desc[df_polarity_desc['Sentiment_Type']=='Negative Review']
negative_reviews.head()
wc(positive_reviews['Review'],'black','Most Used Words')
wc(negative_reviews['Review'],'black','Most Used Words')
import string

def text_process(review):

    nopunc=[word for word in review if word not in string.punctuation]

    nopunc=''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
test_df=test_df.dropna(axis=0,how='any')

rating_class = test_df[(test_df['Rating'] == 1) | (test_df['Rating'] == 10)]

X_review=rating_class['Review']

y=rating_class['Rating']
len(X_review)
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer= CountVectorizer(analyzer=text_process).fit(X_review)
X_review = bow_transformer.transform(X_review)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_review, y, test_size=0.3, random_state=101)
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train, y_train)

predict=nb.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(confusion_matrix(y_test, predict))

print('\n Accuracy:')

print(accuracy_score(y_test, predict))

print(classification_report(y_test, predict))
rating_positive=test_df['Review'][6]

rating_positive
rating_postive_transformed = bow_transformer.transform([rating_positive])

nb.predict(rating_postive_transformed)[0]
rating_negative=test_df['Review'][54]

rating_negative
rating_negative_transformed = bow_transformer.transform([rating_negative])

nb.predict(rating_negative_transformed)[0]
ratings_1 = (rating_class['Rating']==1).sum()

ratings_1_indices = np.array(rating_class[rating_class.Rating == 1].index)

ratings_10_indices = rating_class[rating_class.Rating == 10].index





random_normal_indices = np.random.choice(ratings_10_indices, ratings_1, replace = False)

random_normal_indices = np.array(random_normal_indices)



under_sample_indices = np.concatenate([ratings_1_indices,random_normal_indices])







undersample = rating_class.ix[under_sample_indices]



X_undersample = undersample.ix[:, undersample.columns != 'Rating']

y_undersample = undersample.ix[:, undersample.columns == 'Rating']
print("Percentage of 10 ratings: ", len(undersample[undersample.Rating == 10])/len(undersample))

print("Percentage of 1 ratings: ", len(undersample[undersample.Rating == 1])/len(undersample))

print("Total number of examples in resampled data: ", len(undersample))
X_review_us = X_undersample['Review']
X_review_us = bow_transformer.transform(X_review_us)
X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(X_review_us, y_undersample, test_size=0.3, random_state=101)
nb.fit(X_train_us, y_train_us)

predict_us=nb.predict(X_test_us)
print(confusion_matrix(y_test_us, predict_us))

print('\n Accuracy:')

print(accuracy_score(y_test_us, predict_us))

print(classification_report(y_test_us, predict_us))
nb.fit(X_train_us, y_train_us)

predict_entire=nb.predict(X_test)
print(confusion_matrix(y_test, predict_entire))

print('\n Accuracy:')

print(accuracy_score(y_test, predict_entire))

print(classification_report(y_test, predict_entire))
print(confusion_matrix(y_test, predict))

print('\n Accuracy:')

print(accuracy_score(y_test, predict))

print(classification_report(y_test, predict))