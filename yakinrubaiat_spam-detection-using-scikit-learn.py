import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/emails.csv", encoding= "latin-1")
data.spam.value_counts()
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(data["text"],data["spam"], test_size=0.2, random_state=10)
from sklearn.feature_extraction.text import CountVectorizer
help(CountVectorizer)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words="english")
vect.fit(train_X) # Find some word that cause most of spam email
print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])
X_train_df = vect.transform(train_X)
X_test_df = vect.transform(test_X)
type(X_test_df)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
model = MultinomialNB(alpha=1.8)
model.fit(X_train_df,train_y)
pred = model.predict(X_test_df)
accuracy_score(test_y, pred)
print(classification_report(test_y, pred , target_names = ["Not Spam", "Spam"]))
confusion_matrix(test_y,pred)
print(data["text"][1472])
pred = model.predict(vect.transform(data["text"]))
print("Pred : ",pred[1472])
print("Main : ",data["spam"][1472])
print(data["text"][10])
pred = model.predict(vect.transform(data["text"]))
print("Pred : ",pred[10])
print("Main : ",data["spam"][10])
dir(vect.transform)