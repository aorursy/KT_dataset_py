#  Firstly Import the mendatory packages like,

# pandas, numpy, seaborn.



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra | numerical array and some statastics



# Feature models

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

#  Generate new data frame from news.csv



df = pd.read_csv('../input/news-data-set-fake-news-with-python/news_datasets.csv') # delimiter=','



# Show first five News (rows) from loaded data.

df.head(5)
# To know what's the shape of our dataframe.

df.shape
# Select all labels ( FAKE | REAL )

labels = df['label']



# print first five labels.

labels.head()


# Split the Trainng and Testing data set

x_train, x_test, y_train, y_test = train_test_split( df.text , labels, test_size=0.30, random_state=42)

 # Apply TfidfVectorizer 

    

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# fitting and transformming  test set



tfidf_train = tfidf_vectorizer.fit_transform(x_train) 

tfidf_test = tfidf_vectorizer.transform(x_test)

# setup  a PassiveAggressiveClassifier



pac = PassiveAggressiveClassifier( max_iter = 50)

pac.fit(tfidf_train, y_train)

# Check Predict on the test set and calculate accuracy value



y_pred = pac.predict(tfidf_test)

score = accuracy_score(y_test,y_pred)

print( 'Model Accuracy :', round(score*100,2) )
# Build the confusion matrics here



confusion_matrix(y_test, y_pred, labels=labels.unique())

print("*"*50, "")

print(" That's all, Thank you I hope you like it and you learn somet nahing from this notebook")

print("*"*50)