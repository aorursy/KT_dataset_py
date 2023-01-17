import pandas as pd
import numpy as np
import seaborn as sns
dataset = pd.read_csv("../input/Restaurant_Reviews.tsv", delimiter='\t',quoting=3)
dataset.head(2)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
dataset.shape

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset["Review"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
x.shape
y = dataset.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.20,random_state=0)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)
#predict
y_pred = classifier.predict(x_test)
from sklearn.metrics import  confusion_matrix,classification_report
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_pred))
#73%accuracy
from sklearn.linear_model import LogisticRegression
classifier2 =LogisticRegression(random_state=0)
classifier2.fit(x_train,y_train)
y_pred2 = classifier2.predict(x_test)
cm = confusion_matrix(y_test,y_pred2)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_pred2))
#71% Accuracy








