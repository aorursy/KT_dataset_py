import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



#READING INPUT

training_data = pd.read_csv("../input/train.csv")

testing_data=pd.read_csv("../input/test.csv")

training_data.head()
training_data['author_num'] = training_data.author.map({'EAP':0, 'HPL':1, 'MWS':2})

X = training_data['text']

y = training_data['author_num']

print (X.head())

print (y.head())



per=int(float(0.7)* len(X))

X_train=X[:per]

X_test=X[per:]

y_train=y[:per]

y_test=y[per:]
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
#toy example

text=["My name is computer My life"]

toy = CountVectorizer(lowercase=False, token_pattern=r'\w+|\,')

toy.fit_transform(text)

print (toy.vocabulary_)

matrix=toy.transform(text)

print (matrix[0,0])

print (matrix[0,1])

print (matrix[0,2])

print (matrix[0,3])

print (matrix[0,4])



vect = CountVectorizer(lowercase=False, token_pattern=r'\w+|\,')

X_cv=vect.fit_transform(X)

X_train_cv = vect.transform(X_train)

X_test_cv = vect.transform(X_test)

print (X_train_cv.shape)
clf=MultinomialNB()

clf.fit(X_train_cv, y_train)

clf.score(X_test_cv, y_test)
X_test=vect.transform(testing_data["text"])





clf=MultinomialNB()

clf.fit(X_cv, y)

predicted_result=clf.predict_proba(X_test)

predicted_result.shape
#NOW WE CREATE A RESULT DATA FRAME AND ADD THE COLUMNS NECESSARY TO SUBMIT HERE

result=pd.DataFrame()

result["id"]=testing_data["id"]

result["EAP"]=predicted_result[:,0]

result["HPL"]=predicted_result[:,1]

result["MWS"]=predicted_result[:,2]

result.head()
result.to_csv("TO_SUBMIT.csv", index=False)