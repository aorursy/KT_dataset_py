import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
F = pd. read_csv('../input/fake-and-real-news-dataset/Fake.csv')
T = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
F['text'].apply(len).mean() 
# Fake News Text length mean.. how long is the text of the news
T['text'].apply(len).mean()
# True News Text length mean.. how long is the text of the news
F['text'].apply(len).std() 
T['text'].apply(len).mean() 
F['title'].apply(len).mean()
# Fake News Title length mean.. how long is the title of the news
T['title'].apply(len).mean()
# True News Title length mean.. how long is the title of the news
# it looks like the title of real news is short.. 
F['title'].apply(len).std()
# trying to figure out how much do they deviate.. 
T['title'].apply(len).std()
# There seems to be an overlap, so cannot move forward with using lengths as a measure of Fake News
F.groupby('subject').count()
#Trying to figure out if the subject matters to determine fake or real
T.groupby('subject').count()
#we can observe a difference
T['labels'] = 1
F['labels'] = 0
df = pd.concat([T,F])
# We will add labels to both the news types to concatanate them and then train them 
df.count() # looking at total rows and columns
# we will be working with only the titles. 
import re # for data cleaning
import nltk# general.. for stop words and stem porter
nltk.download('stopwords')# stop words part 1
from nltk.corpus import stopwords # stopwords part 2
from nltk.stem.porter import PorterStemmer # love and loved will be characterized as one word.. 
#so we need this
import string # we will try to remove punctuation by this method
nopunc = [c for c in df['title'] if c not in string.punctuation]
# if we directly try to enter df['title'], it gives a value error. 
corpus = []
for i in range(0,44898):
    title = re.sub('[^a-zA-Z]', ' ', nopunc[i]) # over here, cannot directly take df['title']
    #this also will remove punctuations and brackets and everything else
    title = title.lower()#lower the uppercase
    title = title.split()#will split each word
    ps = PorterStemmer()
    title = [ps.stem(words) for words in title if not words in set(stopwords.words('english'))]
    #stopwords will be removed after splitting
    title = ' '.join(title)#we join the remaining words back
    corpus.append(title)#whatever is left we put it in the corpus
    
#This happens for 44898 times for every news.
from sklearn.model_selection import train_test_split# trainign and testing 
X_train, X_test, y_train, y_test = train_test_split(corpus, df['labels'], test_size = 0.20, random_state = 0)
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
pipeline = Pipeline([('bow', CountVectorizer()), ('classifier', MultinomialNB())])
pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))
cv = CountVectorizer() # max_features = 13000 ---> we can add this argument(optional)
X=cv.fit_transform(corpus).toarray()
y= df['labels'].values
len(X[0])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
prediction2 = classifier.predict(X_test)
print(classification_report(y_test, prediction2))