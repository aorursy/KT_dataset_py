import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/spamraw.csv')

df.head()
sns.countplot(data = df, x= df["type"]).set_title("Amount of spam and no-spam messages", fontweight = "bold")

plt.show()
from sklearn.model_selection import train_test_split



# train_test_split (X, Y, test_size=0.2, random_state=0)

data_train, data_test, labels_train, labels_test = train_test_split(

    df.text,df.type,test_size=0.2,random_state=0) 

from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()



#fit & transform

# fit: build dict (i.e. word->wordID)  

# transform: convert document (i.e. each line in the file) to word vector 

data_train_count = vectorizer.fit_transform(data_train)

data_test_count  = vectorizer.transform(data_test)
from sklearn.naive_bayes import MultinomialNB



clf = MultinomialNB()

clf.fit(data_train_count, labels_train)

predictions = clf.predict(data_test_count)

print(predictions)
from sklearn.metrics import accuracy_score



print (accuracy_score(labels_test, predictions))
from sklearn.metrics import classification_report,confusion_matrix

print (confusion_matrix(labels_test, predictions))
print (classification_report(labels_test, predictions))