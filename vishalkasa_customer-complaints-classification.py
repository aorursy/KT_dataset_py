import numpy as np
import pandas as pd
import tensorflow as tf
data=pd.read_csv("../input/consumer_complaints/Consumer_Complaints.csv")
data.head()
data.columns
data.isnull().sum()
data.info()
np.random.seed(10)
remove_n = 900000
drop_indices = np.random.choice(data.index, remove_n, replace=False)
data = data.drop(drop_indices)
data.shape
new=["Consumer Complaint","Product"]
newdata=data[new]
newdata.describe()
newdata=newdata.drop_duplicates()
newdata=newdata.dropna(axis=0)
newdata.info()
newdata['Product'] = newdata['Product'].astype('category')
newdata["Id"] = newdata['Product'].cat.codes

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

lemmatizer = WordNetLemmatizer() 
tokenizer = RegexpTokenizer(r'\w+')

def sentence_clean(sentence):
    word_list = tokenizer.tokenize(sentence)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(newdata['Consumer Complaint']).toarray()
labels = newdata["Id"]
features.shape
X_train, X_test, y_train, y_test = train_test_split(newdata['Consumer Complaint'], newdata['Product'],test_size=0.3,random_state=2)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
reg = LogisticRegression(max_iter=500,multi_class="auto",penalty='l2')
reg.fit(X_train_tfidf, y_train)
acc_log_train = round(reg.score(X_train_tfidf, y_train) * 100, 2)
acc_log_test = round(reg.score(count_vect.transform(X_test), y_test) * 100, 2)
random_forest = RandomForestClassifier(n_estimators=500)
random_forest.fit(X_train_tfidf, y_train)
acc_random_forest = round(random_forest.score(X_train_tfidf, y_train) * 100, 2)
acc_random_forest_test = round(random_forest.score(count_vect.transform(X_test), y_test) * 100, 2)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train_tfidf, y_train) 
acc_decision_tree = round(decision_tree.score(X_train_tfidf, y_train) * 100, 2)
acc_decision_tree_test = round(decision_tree.score(count_vect.transform(X_test), y_test) * 100, 2)
test=KNeighborsClassifier(n_neighbors=500)
test.fit(X_train_tfidf, y_train) 
acc_Kneighbour=test.score(X_train_tfidf, y_train) * 100
acc_Kneighbour_test=test.score(count_vect.transform(X_test), y_test) * 100
linear_svc = LinearSVC( penalty='l1',dual=False,tol=1e-5)
linear_svc.fit(X_train_tfidf, y_train) 
acc_linear_svc = round(linear_svc.score(X_train_tfidf, y_train) * 100)
acc_linear_svc_test = round(linear_svc.score(count_vect.transform(X_test), y_test) * 100)
import matplotlib.pyplot as plt
Model=["RandomForestClassifier","DecisionTreeClassifier","KNeighborsClassifier","LogisticRegression","SVM"]
Accuracy=[acc_random_forest,acc_decision_tree,acc_Kneighbour,acc_log_train,acc_linear_svc]
plt.barh(Model,Accuracy,color="r")
for index, value in enumerate(Accuracy):
    plt.text(value, index, str(value))

Model=["RandomForestClassifier","DecisionTreeClassifier","KNeighborsClassifier","LogisticRegression","SVM"]
Accuracy=[acc_random_forest_test,acc_decision_tree_test,acc_Kneighbour_test,acc_log_test,acc_linear_svc_test]
plt.barh(Model,Accuracy)
for index, value in enumerate(Accuracy):
    plt.text(value, index, str(value))
