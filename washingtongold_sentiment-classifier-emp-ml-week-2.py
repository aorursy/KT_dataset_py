# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv')
data = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1')
data.head()
data.columns
data = data.drop(['1467810369', 'Mon Apr 06 22:19:45 PDT 2009', 'NO_QUERY', '_TheSpecialOne_'],axis=1)
data.head()
data = data.rename(columns={'0':'Sentiment', 

                            "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D":'Text'})
data.head()
def add1(value):

    return value+1

data['Sentiment'].apply(add1)
data['Sentiment'] = data['Sentiment'].apply(add1)
data.head()
def clean_text(text):

    #First, let's make the text lowercase.

    text = text.lower()

    

    #Next, let's remove exclamation marks.

    text_list = text.split('!') #This returns a list of items, split by !.

    text = ''.join(text_list) #This puts the list back into a string. The exclamation marks are gone.

    

    #Same drill, but with other types of punctuation.

    #For each punctuation mark in this list...

    for punctuation in ['.','?',':','@','#','$','%','^','&','*','(',')','"',"'",",",";",'[',']']:

        #Split the text by that punctuation mark

        text_list = text.split(punctuation)

        #And rejoin it!

        text = ''.join(text_list)

    

    return text
clean_text('! jijd.? SOIDJOIS.SJOIJ(^*(SJIJj.s!!io))')
data = data[ data['Text'].apply(len) < 25 ]

data
data['Cleaned'] = data['Text'].apply(clean_text)
data.head()
data = data.reset_index()

data
data = data.drop('index',axis=1)

data
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer() #Create an instance, as usual



vectorizer.fit(data['Cleaned']) #Like any sklearn model, it needs to be fit.
transformed = vectorizer.transform(data['Cleaned']) #Then, we can use the vectorizer to 'transform', or vectorize the cleaned data.

#We can store the data into variable called 'transformed'.
transformed.shape
X = transformed #These are the 'features', the X, that we will be plugging into the model...

y = data['Sentiment'] #...and we want to get a sentiment rating from 1 to 5 out.
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.6,stratify=y) 

#Just to speed up training time. We normally wouldn't want to make the test larger than the train.



#Stratifying is simply making sure that there is an equal amount of a class in a training set (e.g. 1k of label 1, 1k of label 2, etc.)

#Prevents overfitting to one specific class.
from sklearn.linear_model import LogisticRegression #import model

log = LogisticRegression() #Create instance

log.fit(X_train,y_train) #Fit
from sklearn.tree import DecisionTreeClassifier

dec = DecisionTreeClassifier()

dec.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

accuracy_score(log.predict(X_test),y_test)
accuracy_score(dec.predict(X_test),y_test)
print("Enter your text.")

text = input(':: ') #Let this be the text input.



#Step 1 of the pipeline: clean the text

text1 = clean_text(text)



#Step 2 of the pipeline: vectorize the cleaned text

text2 = vectorizer.transform([text1])



#Step 3 of the pipeline: get a prediction for the sentiment.

prediction = log.predict(text2)



print("Sentiment:",prediction)



#Try out: 'this is horrible' 'omg i love you'