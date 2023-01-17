import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")
df.tail()
df.info()
df.drop(df[df['Review Text'].isna()].index,inplace=True) #drop where there are no text
blanks = []  # start with an empty list



for i,lb,rv in df[['Review Text','Title']].itertuples():  # iterate over the DataFrame

    if type(rv)==str:            # avoid NaN values

        if rv.isspace():         # test 'review' for whitespace

            blanks.append(i)
blanks #there are no blanks or space instead of NaN
df.info()
#df[df['Rating']==3]
df['Title']=df['Title'].apply(lambda x:" " if pd.isnull(x) else x) #replace null value with a space
df.info()
#df[df['Division Name'].isna()]
df['Division Name'].fillna(df['Division Name'].mode()[0],inplace=True) # replace nan with most common value that occur
df['Department Name'].fillna(df['Department Name'].mode()[0],inplace=True)
df['Class Name'].fillna(df['Class Name'].mode()[0],inplace=True)
df.info() # data types are fine plus there are no null values left
df.head()
df['Title-Review Text']=df[['Title', 'Review Text']].apply(lambda x: ' '.join(x), axis=1)
df.head(6)
df.drop("Unnamed: 0",axis=1,inplace=True)
df['Review']=df['Rating'].apply(lambda x: "positive" if x>3 else("negative" if x<3 else("neutral" if x==3 else x)))
df.head()
df=df.sort_values("Clothing ID")

df.reset_index(drop=True,inplace=True)
df.iloc[18111:,:]['Clothing ID'].unique()
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
#import nltk

#nltk.download("stopwords")

#nltk.download('punkt')
"""

Reference from Ken Jee : https://github.com/PlayingNumbers/ds_salary_proj

"""

words = " ".join(df['Title-Review Text'][df['Review']=="positive"])



def punctuation_stop(text):

    """remove punctuation and stop words"""

    filtered = []

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(text)

    for w in word_tokens:

        if w not in stop_words and w.isalpha():

            filtered.append(w.lower())

    return filtered





words_filtered = punctuation_stop(words)





text = " ".join([ele for ele in words_filtered])



wc= WordCloud(background_color="white", random_state=1,stopwords=STOPWORDS, max_words = 2000, width =1000, height = 1500)

wc.generate(text)



plt.figure(figsize=[10,10])

plt.imshow(wc,interpolation="bilinear")

plt.axis('off')

plt.show()
words = " ".join(df['Title-Review Text'][df['Review']=="negative"])



words_filtered = punctuation_stop(words)





text = " ".join([ele for ele in words_filtered])



wc= WordCloud(background_color="white", random_state=1,stopwords=STOPWORDS, max_words = 2000, width =1000, height = 1500)

wc.generate(text)



plt.figure(figsize=[10,10])

plt.imshow(wc,interpolation="bilinear")

plt.axis('off')

plt.show()
words = " ".join(df['Title-Review Text'][df['Review']=="neutral"])



words_filtered = punctuation_stop(words)





text = " ".join([ele for ele in words_filtered])



wc= WordCloud(background_color="white", random_state=1,stopwords=STOPWORDS, max_words = 2000, width =1000, height = 1500)

wc.generate(text)



plt.figure(figsize=[10,10])

plt.imshow(wc,interpolation="bilinear")

plt.axis('off')

plt.show()
sns.barplot(x="Review",y="Age",data=df)
sns.barplot(x="Review",y="Recommended IND",data=df)
sns.barplot(x="Review",y="Positive Feedback Count",data=df)
df.reset_index(drop=True,inplace=True)
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import re

ps = PorterStemmer()

corpus = []

for i in range(0, len(df)):

    review = re.sub('[^a-zA-Z]', ' ', df.loc[i,'Title-Review Text'])

    review = review.lower()

    review = review.split()

    

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)
df['Title-Review Text']=corpus
df.columns
df['Review']=df['Review'].apply(lambda x:0 if x=="negative" else(2 if x=='positive' else(1 if x=='neutral' else x)))
#df=df.sort_values("Clothing ID")

#df.reset_index(drop=True,inplace=True)
words=df['Title-Review Text']

y=df['Review']
#words
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_v=TfidfVectorizer(max_features=3000,ngram_range=(1,3))

words=tfidf_v.fit_transform(words).toarray()
X_train=words[:18111]

X_test=words[18111:]

y_train=y[:18111].values

y_test=y[18111:].values
#pd.DataFrame(words).to_csv("words.csv")
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import classification_report

print("Classification Report:\n ", classification_report(y_test, y_pred))
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import classification_report

print("Classification Report:\n ", classification_report(y_test, y_pred))
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(class_weight="balanced")

classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import classification_report

print("Classification Report:\n ", classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import classification_report

print("Classification Report:\n ", classification_report(y_test, y_pred))
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import classification_report

print("Classification Report:\n ", classification_report(y_test, y_pred))
#from imblearn.combine import SMOTETomek
#smk = SMOTETomek(random_state=42)
#X_train_res,y_train_res=smk.fit_sample(X_train,y_train)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(class_weight="balanced")

classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import classification_report

print("Classification Report:\n ", classification_report(y_test, y_pred))
#import pickle 

#pickle.dump(classifier,open("model.pkl","wb"))
#from sklearn.pipeline import Pipeline

#from sklearn.feature_extraction.text import TfidfVectorizer

#from sklearn.linear_model import LogisticRegression

#text_clf_nb = Pipeline([('tfidf', TfidfVectorizer(max_features=3000,ngram_range=(1,3))),

#                     ('clf', LogisticRegression(class_weight="balanced")),

#])

#text_clf_nb.fit(X_train,y_train)
#import pickle 

#pickle.dump(text_clf_nb,open("model1.pkl","wb"))
#text_clf_nb.predict(X_test)
#df=pd.read_csv("women-clothing.csv")

#load=pickle.load(open('model1.pkl','rb'))