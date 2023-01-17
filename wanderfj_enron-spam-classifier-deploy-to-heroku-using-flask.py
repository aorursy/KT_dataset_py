# Carregamento dos dados

import numpy as np

import pandas as pd

from sklearn.datasets import load_files



X, y = [], []

for i in range(1,7):

    emails = load_files(f"../input/enron-spam/enron{i}")

    X = np.append(X, emails.data)

    y = np.append(y, emails.target)



classes = emails.target_names
# Verificação dos dados - tamanho e imprimir exemplo

print(f"X.shape: {X.shape:}")

print(f"y.shape: {y.shape}")

print("\n")

print(f"Exemplo X[0]: {X[0]}")

print("\n")

print(f"Classe X[0]: {y[0]} ({classes[int(y[0])]})")
# Verificação dos dados - distribuição do target (classes balanceadas)



from collections import Counter

import matplotlib.pyplot as plt

%matplotlib inline



plt.bar(Counter(y).keys(), Counter(y).values(),tick_label =('spam', 'ham'))

plt.show;
# Limpeza dos dados

import re



X_tratado = []



for email in range(0, len(X)): 

    

    # Remover caracteres especiais

    texto = re.sub(r'\\r\\n', ' ', str(X[email]))

    texto = re.sub(r'\W', ' ', texto)

    

    # Remove caracteres simples de uma letra

    texto = re.sub(r'\s+[a-zA-Z]\s+', ' ', texto)

    texto = re.sub(r'\^[a-zA-Z]\s+', ' ', texto) 



    # Substitui multiplos espaços por um unico espaço

    texto = re.sub(r'\s+', ' ', texto, flags=re.I)



    # Remove o 'b' que aparece no começo

    texto = re.sub(r'^b\s+', '', texto)



    # Converte para minúsculo

    texto = texto.lower()



    X_tratado.append(texto)
print(f"Exemplo X[0]: {X_tratado[0]}")

print("\n")

print(f"Classe X[0]: {y[0]} ({classes[int(y[0])]})")
# Separa os dados em conjunto de treinamento e teste

from sklearn.model_selection import train_test_split  



X_train, X_test, y_train, y_test = train_test_split(X_tratado, y, test_size=0.3)
# Aplicação do classificador Naive Bayes

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score



clf_pipeline = Pipeline([

    ('tfidf_vectorizer', TfidfVectorizer()),

    ('classificador', MultinomialNB())])



clf_pipeline.fit(X_train, y_train)

predictions = clf_pipeline.predict(X_train)

score = accuracy_score(y_train,predictions)

print(f"Acurácia de treinamento: {score*100:.2f}%")



predictions_test = clf_pipeline.predict(X_test)

score = accuracy_score(y_test, predictions_test)

print(f"Acurácia de teste: {score*100:.2f}%")

print()



print("Matriz de confusão do set de teste:")

y_true = pd.Series(y_test, name='Real')

y_pred = pd.Series(predictions_test, name='Previsto')

pd.crosstab(y_true, y_pred)
# Teste de previsão em um texto novo de e-mail:



texto = """Folks.

I have a meeting today to discuss a research project, and then I was invited to a meeting with the campus 

board about continuing the Data Science specialization course.

This second meeting is expected to start at 5 pm and will likely extend until after our class start time.

That way, I think it's best to release you from today's class.

Today's content is “data wrangling,” which corresponds to chapters 5 through 8 of the book Python for 

Data Analysis, which is part of the file I sent you earlier this semester. Please read these chapters 

(they are short) and do the exercises during this week.

In our next class we will start with a section of questions about these materials and then we will talk 

about feature selection selection.

See you on the 31st, then.

Regards, Jefferson Andrade."""



previsao = clf_pipeline.predict([texto])

probabilidade  = np.max(clf_pipeline.predict_proba([texto]))



print(f"Classe prevista: {previsao} = {classes[int(previsao)]} com probabilidade {probabilidade*100:.2f}%")
# Função que receba o nome do arquivo com a mensagem e informe se a mensagem é ou não spam



def prever_spam_ham(caminho):

    arquivo = open(caminho,"r", encoding="utf8") 

    texto = arquivo.read()

    arquivo.close()

    previsao = clf_pipeline.predict([texto])

    probabilidade  = np.max(clf_pipeline.predict_proba([texto]))

    print(f"Texto: {texto}")

    print("\n")

    print(f"Classe prevista: {previsao} = {classes[int(previsao)]} com probabilidade {probabilidade*100:.2f}%")

    

# Usage

# prever_spam_ham("texto.txt")
# Salvar modelo em Joblib

import joblib



joblib.dump(clf_pipeline, 'modelo.pkl')