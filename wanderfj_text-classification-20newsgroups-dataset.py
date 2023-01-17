# Carregamento do dataset 20 Newsgroups



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.datasets import fetch_20newsgroups



newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))



X_train = np.array(newsgroups_train.data)

y_train = np.array(newsgroups_train.target)

X_test = np.array(newsgroups_test.data)

y_test = np.array(newsgroups_test.target)
# EDA - Análise Exploratória de Dados



#Verficação de um dos exemplos



print(X_train[0])
newsgroups_train.target_names[y_train[0]]
# Contagem de documentos de treino e teste por label



def conta_labels(y_train, y_test):

    """Returna dataframe com os total de documentos em cada classe

    de treinamento e teste. Ref.: Cachopo (2007)"""

    y_train_classes = pd.DataFrame([newsgroups_train.target_names[i] for i in newsgroups_train.target])[0]

    y_test_classes = pd.DataFrame([newsgroups_test.target_names[i] for i in newsgroups_test.target])[0]

    

    contagem_df = pd.concat([y_train_classes.value_counts(),

                             y_test_classes.value_counts()],

                            axis=1, 

                            keys=["# docs treino", "# docs teste"], 

                            sort=False)

    

    contagem_df["# total docs"] = contagem_df.sum(axis=1)

    contagem_df.loc["Total"] = contagem_df.sum(axis=0)

    

    return contagem_df



newsgroups_df_labels = conta_labels(y_train, y_test)

newsgroups_df_labels
# Exibição no formato de gráfico

%matplotlib inline



newsgroups_df_labels.iloc[:-1,:-1].plot.barh(stacked=True, 

                                    figsize=(10, 8),

                                    title="Número de documentos de treinamento e teste por classe");
# Carregamento das bibliotecas comuns do sklearn



from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import metrics
# Classe de Pre-Processamento de textos utilizando a biblioteca NLTK



import string

import re

import nltk



class NLTKTokenizer():

    """Classe que recebe documentos como entrada e devolve realizado lematização

    e retirando stopwords e pontuacoes.

    Ref.: https://scikit-learn.org/stable/modules/feature_extraction.html

    """    

    def __init__(self):

        self.lemmatizer = nltk.stem.WordNetLemmatizer()

        self.stopwords = nltk.corpus.stopwords.words('english')

        self.english_words = set(nltk.corpus.words.words())

        self.pontuacao = string.punctuation



    def __call__(self, doc):

        # ETAPA 1 - Limpeza de texto

        # Converte para minúsculo

        doc = doc.lower()       

        

        # Trocar numeros pela string numero

        doc = re.sub(r'[0-9]+', 'numero', doc)

        

        # Trocar underlines por underline

        doc = re.sub(r'[_]+', 'underline', doc)



        # Trocar URL pela string httpaddr

        doc = re.sub(r'(http|https)://[^\s]*', 'httpaddr', doc)

        

        # Trocar Emails pela string emailaddr

        doc = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', doc) 

        

        # Remover caracteres especiais

        doc = re.sub(r'\\r\\n', ' ', doc)

        doc = re.sub(r'\W', ' ', doc)



        # Remove caracteres simples de uma letra

        doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc)

        doc = re.sub(r'\^[a-zA-Z]\s+', ' ', doc) 



        # Substitui multiplos espaços por um unico espaço

        doc = re.sub(r'\s+', ' ', doc, flags=re.I)

        

        # ETAPA 2 - Tratamento da cada palavra

        palavras = []

        for word in nltk.word_tokenize(doc):

            if word in self.stopwords:

                continue

            if word in self.pontuacao:

                continue

            if word not in self.english_words:

                continue

            

            word = self.lemmatizer.lemmatize(word)

            palavras.append(word)

        

        return palavras
# Vetores de características sem tratamento com NLTK e Expressões Regulares



vetorizador = CountVectorizer()

v1 = vetorizador.fit_transform(X_train)



features = vetorizador.get_feature_names()

v1_df = pd.DataFrame(v1.toarray(), columns = features)

v1_df
# Vetores de características com NLTK (lematização, remoção de stopwords e palavras desconhecidas)



vetorizador_tratado = CountVectorizer(tokenizer=NLTKTokenizer())

v2 = vetorizador_tratado.fit_transform(X_train)



features = vetorizador_tratado.get_feature_names()

v2_df = pd.DataFrame(v2.toarray(), columns = features)

v2_df
# Regressão Logística



from sklearn.linear_model import LogisticRegression



text_clf_logistic_regression = Pipeline([('vect', CountVectorizer(tokenizer=NLTKTokenizer())),

                     ('tfidf', TfidfTransformer()),

                     ('clf', LogisticRegression(penalty='l2', 

                                                dual=False, 

                                                tol=0.001, 

                                                C=1.0, 

                                                fit_intercept=True, 

                                                intercept_scaling=1, 

                                                class_weight=None, 

                                                random_state=None, 

                                                solver='lbfgs', 

                                                max_iter=1000, 

                                                multi_class='multinomial', 

                                                verbose=0, 

                                                warm_start=False, 

                                                n_jobs=None, 

                                                l1_ratio=None)),

                     ])



#text_clf_logistic_regression.fit(X_train, y_train)

#predicted = text_clf_logistic_regression.predict(X_test)

#print(metrics.classification_report(y_test, predicted))
# Naive Bayes



from sklearn.naive_bayes import MultinomialNB



text_clf_naive_bayes = Pipeline([('vect', CountVectorizer(tokenizer=NLTKTokenizer())),

                     ('tfidf', TfidfTransformer()),

                     ('clf', MultinomialNB(alpha=1.0, 

                                           fit_prior=True, 

                                           class_prior=None)),

                     ])



#text_clf_naive_bayes.fit(X_train, y_train)

#predicted = text_clf_naive_bayes.predict(X_test)

#print(metrics.classification_report(y_test, predicted))
# KNN



from sklearn.neighbors import KNeighborsClassifier



text_clf_knn = Pipeline([('vect', CountVectorizer(tokenizer=NLTKTokenizer())),

                     ('tfidf', TfidfTransformer()),

                     ('clf', KNeighborsClassifier(n_neighbors=5, 

                                                  weights='uniform', 

                                                  algorithm='auto', 

                                                  leaf_size=30, 

                                                  p=2, 

                                                  metric='minkowski', 

                                                  metric_params=None, 

                                                  n_jobs=None)),

                     ])



#text_clf_knn.fit(X_train, y_train)

#predicted = text_clf_knn.predict(X_test)

#print(metrics.classification_report(y_test, predicted))
# Árvore de Decisão



from sklearn.tree import DecisionTreeClassifier



text_clf_decision_tree = Pipeline([('vect', CountVectorizer(tokenizer=NLTKTokenizer())),

                         ('tfidf', TfidfTransformer()),

                         ('clf', DecisionTreeClassifier(criterion='gini', 

                                                        splitter='best', 

                                                        max_depth=None, 

                                                        min_samples_split=2, 

                                                        min_samples_leaf=1, 

                                                        min_weight_fraction_leaf=0.0, 

                                                        max_features=None, 

                                                        random_state=None, 

                                                        max_leaf_nodes=None, 

                                                        min_impurity_decrease=0.0, 

                                                        min_impurity_split=None, 

                                                        class_weight=None, 

                                                        presort=False)),

                         ])



#text_clf_decision_tree.fit(X_train, y_train)

#predicted = text_clf_decision_tree.predict(X_test)

#print(metrics.classification_report(y_test, predicted))
# Random Forest



from sklearn.ensemble import RandomForestClassifier



text_clf_rf = Pipeline([('vect', CountVectorizer(tokenizer=NLTKTokenizer())),

                     ('tfidf', TfidfTransformer()),

                     ('clf', RandomForestClassifier(n_estimators=100, 

                                                    criterion='gini', 

                                                    max_depth=None, 

                                                    min_samples_split=2, 

                                                    min_samples_leaf=1, 

                                                    min_weight_fraction_leaf=0.0, 

                                                    max_features='auto', 

                                                    max_leaf_nodes=None, 

                                                    min_impurity_decrease=0.0, 

                                                    min_impurity_split=None, 

                                                    bootstrap=True, 

                                                    oob_score=False, 

                                                    n_jobs=None, 

                                                    random_state=None, 

                                                    verbose=0, 

                                                    warm_start=False, 

                                                    class_weight=None)),

                     ])



#text_clf_rf.fit(X_train, y_train)

#predicted = text_clf_rf.predict(X_test)

#print(metrics.classification_report(y_test, predicted))
# Classe avaliadora de performance dos classificadores (com base no exemplo do Prof. Boldt)



from sklearn.model_selection import KFold



class PerformanceEvaluator():

    """ Classe avaliadora de performance dos classificadores (com base no exemplo do Prof. Boldt) """

    def __init__(self, X_train, y_train):

        self.X_train = X_train

        self.y_train = y_train

        self.kf = KFold(n_splits=5)

    

    def score(self, clf):

        scores = []

        for train, validate in self.kf.split(self.X_train):

            clf.fit(self.X_train[train],self.y_train[train])

            scores.append(clf.score(self.X_train[validate],self.y_train[validate]))

        return np.mean(scores), np.std(scores)

    

    def treinar(self, clfs):

        print(f'{"":>20}  Média \t Desvio Padrão')

        for name,clf in clfs:

            score_mean, score_std = self.score(clf)

            print(f'{name:>20}: {score_mean:.4f} \t {score_std:.4f}')



    def testar(self, clfs, X_test, y_test):

        # Testa os classificadores em dados de teste (não vistos no treinamento)

        for name,clf in clfs:

            score = clf.score(X_test, y_test)

            print(f'{name:>20}: {score:.4f}')
# Avaliação de todos os classificadores



clfs = [

    ('Logistic Regression', text_clf_logistic_regression),

    ('Naive Bayes', text_clf_naive_bayes),

    ('KNN', text_clf_knn),

    ('Decision Tree', text_clf_decision_tree),

    ('Random Forest', text_clf_rf),

]



pe = PerformanceEvaluator(X_train,y_train)
%%time

# Treina os classificadores usando validação cruzada



pe.treinar(clfs)
%%time

# Testa os classificadores em dados de teste (não vistos no treinamento)



pe.testar(clfs, X_test, y_test)