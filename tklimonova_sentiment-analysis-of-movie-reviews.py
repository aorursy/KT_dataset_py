import pandas as pd

import matplotlib.pyplot as plt

import nltk

import numpy as np

import re



from nltk.corpus import stopwords, wordnet 

from nltk import word_tokenize, WordNetLemmatizer, sent_tokenize

from sklearn.model_selection import train_test_split 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

from gensim.models import Word2Vec

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv("../input/imdb-reviews/dataset.csv", encoding="latin1", chunksize=25000)

data_df = next(data)

data_df.head(10)
data_df.duplicated().sum()
data_df.drop_duplicates(keep='first', inplace=True)

data_df.shape
#Are positive reviews longer?

data_df['length_text'] = data_df.SentimentText.str.len()

negative_df = data_df[data_df.Sentiment == 0]

positive_df = data_df[data_df.Sentiment == 1]



plt.hist(negative_df.length_text, bins=20, alpha=0.6, label='negative')

plt.hist(positive_df.length_text, bins=20, alpha=0.6, label='positive')

plt.legend(loc='upper right')

plt.xlabel("Length of review (symbols)")

plt.ylabel("Frequency")

plt.show()
data_df['SentimentText_clean'] = data_df.SentimentText.str.lower()
def clean_str(string):

    

    string = re.sub(r'\<a href', ' ', string)

    string = re.sub(r'&amp;', '', string) 

    string = re.sub(r'<br />', ' ', string)

    string = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', string)

    string = re.sub('\d','', string)

    string = re.sub(r"can\'t", "cannot", string)

    string = re.sub(r"it\'s", "it is", string)

    string = re.sub(r"don\'t", "do not", string)

    string = re.sub(r"i\'d", "i would", string)

    string = re.sub(r"isn\'t", "is not", string)

    string = re.sub(r"wasn\'t", "was not", string)

    string = re.sub(r"i\'m", "i am", string)

    return string
data_df['SentimentText_clean'] = data_df['SentimentText_clean'].apply(clean_str)
#calculating amount of words in review

def word_count(string):

    words = string.split()

    return len(words)

data_df['sum_words'] = data_df['SentimentText_clean'].apply(word_count)
# calculate the most common words

pd.Series(' '.join(data_df['SentimentText_clean']).split()).value_counts().head(10)
#cleaning unrelated words

stop_words = stopwords.words('english') + ['movie', 'film', 'time', 'story', 'cinema', ]

stop_words = set(stop_words)

#remove_stop_words

remove_stop_words = lambda r: [[word for word in word_tokenize(sente) if word not in stop_words] for sente in sent_tokenize(r)]

data_df['SentimentText_tokens'] = data_df['SentimentText_clean'].apply(remove_stop_words)
#finding similar words by sentiment

model = Word2Vec(

        data_df['SentimentText_tokens'].apply(lambda x: x[0]),

        iter=10,

        size=16,

        window=5,

        min_count=5,

        workers=10)
model.wv.most_similar('fun')
#lemmatizing words for better analysis

lemmatizer = WordNetLemmatizer()

wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}



def lemmatize_text(text):

    pos_tag_text = nltk.pos_tag(text.split())

    return ' '.join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tag_text])



data_df['lemmatized_reviews'] = data_df['SentimentText_clean'].apply(lambda text: lemmatize_text(text))

data_df.head()
#Vectorizing data with CountVectorizer



y = data_df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(data_df['lemmatized_reviews'], y, test_size=0.33, random_state=53)

my_stop_words = ENGLISH_STOP_WORDS.union(['aaa', 'aa', 'want', 'make', 'lots','made', 'film', 'time', 'movie', 'films', 'movies', 'cinema', 'story', 'th'])



vect = CountVectorizer(min_df=3, ngram_range=(1, 3), stop_words=my_stop_words)

#creating bag of words for train and test data

count_train = vect.fit_transform(X_train.values)

count_test = vect.transform(X_test.values)



# Create a DataFrame from the bow representation

count_df = pd.DataFrame(count_train.toarray(), columns=vect.get_feature_names())

count_df.head()
#Adding Sentiment column to new DataFrame

count_df['Sentiment'] = data_df.Sentiment
#Naive Bayes model (bag of words vectorizer)

from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB



nb_classifier = MultinomialNB()

nb_classifier.fit(count_train, y_train)



#predicted tags

pred = nb_classifier.predict(count_test)



#accuracy score

score = metrics.accuracy_score(y_test, pred)

print(score)



#confusion matrix

cm = metrics.confusion_matrix(y_test, pred, labels = [0,1])

print(cm)
# Let's test our model with random reviews

review_1 = "The movie was terrible. Even so the music was not bad, I would not reccomend to watch it."

prediction_1 = nb_classifier.predict(vect.transform([review_1]))



review_2 = "Amazing actors play and a lot of humor. I really like this movie and would definetly watch it again."

prediction_2 = nb_classifier.predict(vect.transform([review_2]))



review_3 = "This was such a horrible and sensitive topic to adapt to the screen but they really made the young men’s story justice."

prediction_3 = nb_classifier.predict(vect.transform([review_3]))



review_4 = "They went through a lot of the backstory too quickly. Overall feels pretty low budget and characters feel like they are from 21th Century. The lore is the only interesting part."

prediction_4 = nb_classifier.predict(vect.transform([review_4]))



print("The sentiment of the review is %i" % (prediction_1))

print("The sentiment of the review is %i" % (prediction_2))

print("The sentiment of the review is %i" % (prediction_3))

print("The sentiment of the review is %i" % (prediction_4))
# Build TfidfVectorizer 



y = data_df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(data_df['lemmatized_reviews'], y, test_size=0.33, random_state=53)



# Define the vectorizer and specify the arguments

my_stop_words = ENGLISH_STOP_WORDS.union(['aaa', 'aa', 'want', 'make', 'lots','made', 'film', 'time', 'movie', 'films', 'movies', 'cinema', 'story', 'th'])



vect_tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=2000, stop_words=my_stop_words)



tfidf_train = vect_tfidf.fit_transform(X_train.values)

tfidf_test = vect_tfidf.transform(X_test.values)



# Transform to a data frame and specify the column names

tfidf_df=pd.DataFrame(tfidf_train.toarray(), columns=vect_tfidf.get_feature_names())
#Adding Sentiment column 

tfidf_df['Sentiment'] = data_df.Sentiment

tfidf_df.head()
#Naive Bayes model (tf-idf vectorizer)

from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB



nb_classifier = MultinomialNB()

nb_classifier.fit(tfidf_train, y_train)



#predicted tags

pred = nb_classifier.predict(tfidf_test)



#accuracy score

score = metrics.accuracy_score(y_test, pred)

print(score)



#confusion matrix

cm = metrics.confusion_matrix(y_test, pred, labels = [0,1])

print(cm)
#Vectorizing data with CountVectorizer

my_stop_words = ENGLISH_STOP_WORDS.union(['aaa', 'aa', 'want', 'make', 'lots','made', 'film', 'time', 'movie', 'films', 'movies', 'cinema', 'story', 'th', 'll', 'song', 'dvd', 'mr'])



vect_lda = CountVectorizer(max_df=.1, max_features=5000, stop_words=my_stop_words)

#creating bag of words for train and test data

X = vect_lda.fit_transform(data_df['lemmatized_reviews'].values)
# LDA

from sklearn.decomposition import LatentDirichletAllocation



lda = LatentDirichletAllocation(learning_method = 'online', n_components=10, random_state=123)

X_topics = lda.fit_transform(X)

X_text_c_feature_names = vect_lda.get_feature_names()



def display_topics(model, feature_names, no_top_words):

    for topic_idx, topic in enumerate(model.components_):

        print ("Topic %d:" % (topic_idx))

        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))



n_top_words = 6

display_topics(lda, X_text_c_feature_names, n_top_words)
#let's look at original reviews in Family topic

family = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(family[:3]):

  print('\nFamily movie #%d:' % (iter_idx +1))

  print(data_df['SentimentText'][movie_idx][:500], '...')