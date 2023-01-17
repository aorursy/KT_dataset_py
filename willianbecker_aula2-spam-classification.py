import numpy as np

np.random.seed(100)



import pandas as pd

dataset = pd.read_csv("../input/dataset-spams/dataset_spams.csv")
dataset.head()
print(dataset['spam'].value_counts())
# preparacao dos dados

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



# tokenizando as sentencas

dataset["text"] = [word_tokenize(word) for word in dataset["text"]]

print(dataset)
# remocao de stopwords

dataset["text"] = dataset["text"].apply(lambda x: [item for item in x if item not in stopwords.words("english")])
print(dataset.head())
# DESAFIO

# remover caracteres numericos, pontuacao e outros termos nao relevantes
dataset["text"] = [' '.join(word) for word in dataset["text"]]
print(dataset.head())
from sklearn import model_selection



X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset["text"], 

                                                                    dataset["spam"], 

                                                                    test_size=0.3)
from sklearn.preprocessing import LabelEncoder



Encoder = LabelEncoder()

y_train = Encoder.fit_transform(y_train)

y_test = Encoder.fit_transform(y_test)
from sklearn.feature_extraction.text import TfidfVectorizer



Tfidf_vect = TfidfVectorizer(max_features=5000)

Tfidf_vect.fit(dataset["text"])

Train_X_Tfidf = Tfidf_vect.transform(X_train)

Test_X_Tfidf = Tfidf_vect.transform(X_test)
Tfidf_vect.vocabulary_
from sklearn import naive_bayes

from sklearn.metrics import accuracy_score



Naive = naive_bayes.MultinomialNB()

Naive.fit(Train_X_Tfidf, y_train)



predictions = Naive.predict(Test_X_Tfidf)
print("Acuracia NB:", accuracy_score(predictions, y_test)*100)