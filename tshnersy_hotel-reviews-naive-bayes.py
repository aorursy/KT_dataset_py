import re, string

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import accuracy_score, precision_score



from sklearn.naive_bayes import BernoulliNB,MultinomialNB



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv")

df
df.isnull().sum()

df.info
dist = data['Rating'].value_counts()

fig = px.bar(df, x=dist.index, y=dist.values,labels={'x':'Ratings','y': 'Counts of Ratings'})

fig.show()
wordcloud = WordCloud(width = 800, height = 800, random_state=1,background_color='black', 

                      colormap='Pastel1').generate(" ".join(df["Review"]))



plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud, interpolation='bilinear') 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
x = df["Review"].copy()

y = df["Rating"].copy()
def clean_text(review):

    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''

    review = review.lower()

    review = re.sub('\[.*?\]', '', review)

    review = re.sub('[%s]' % re.escape(string.punctuation), '', review)

    review = re.sub('\w*\d\w*', '', review)

    review = re.sub('[‘’“”…]', '', review)

    review = re.sub('\n', '', review)

    stop_words=set(stopwords.words('english'))

    return review





round1 = lambda x: clean_text(x)

df_clean = x.apply(round1)
df_clean
length = [len(x.split(" ")) for x in df_clean]

fig = px.histogram(length)

fig.show()
tfidfconverter = TfidfVectorizer(max_features=400, min_df=0.05, max_df=0.9)

tfidf = tfidfconverter.fit_transform(df_clean).toarray()
X_train,X_test,y_train,y_test = train_test_split(tfidf,y,test_size=0.2,random_state=42)
nb = MultinomialNB()

nb_bern = BernoulliNB()
nb_model=nb.fit(X_train,y_train)

nb_bern_model=nb_bern.fit(X_train,y_train)
y_pred=nb_model.predict(X_test)

y_pred2=nb_bern_model.predict(X_test)
print("Accuracy Multinominal:",accuracy_score(y_test, y_pred))

print("Accuracy Bernoulli:",accuracy_score(y_test, y_pred2))