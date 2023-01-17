#Importando as bibliotecas

import numpy as np

import pandas as pd

from sklearn.model_selection import KFold
dfTitanic = pd.read_csv('../input/lab5_train_no_nulls_no_outliers_ohe.csv')

dfTitanic.head(3)
# aqui montamos a matriz de atributos X e o vetor coluna de respostas Y.

# Note que não selecionamos algumas colnas, como Nome e Ticket

y = dfTitanic['Survived'].values

X = dfTitanic[['Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S', '1', '2', '3', 'female', 'male']].values
# Dividindo os dados em 5 folds.

kf = KFold(n_splits=5, shuffle=True, random_state=5)
#Função idêntica à usada nos modelos de regressão.

def avalia_classificador(clf, kf, X, y, f_metrica):

    metrica_val = []

    metrica_train = []

    for train, valid in kf.split(X,y):

        x_train = X[train]

        y_train = y[train]

        x_valid = X[valid]

        y_valid = y[valid]

        clf.fit(x_train, y_train)

        y_pred_val = clf.predict(x_valid)

        y_pred_train = clf.predict(x_train)

        metrica_val.append(f_metrica(y_valid, y_pred_val))

        metrica_train.append(f_metrica(y_train, y_pred_train))

    return np.array(metrica_val).mean(), np.array(metrica_train).mean()
def apresenta_metrica(nome_metrica, metrica_val, metrica_train, percentual = False):

    c = 100.0 if percentual else 1.0

    print('{} (validação): {}{}'.format(nome_metrica, metrica_val * c, '%' if percentual else ''))

    print('{} (treino): {}{}'.format(nome_metrica, metrica_train * c, '%' if percentual else ''))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score

lr = LogisticRegression(solver='liblinear')
media_acuracia_val, media_acuracia_train = avalia_classificador(lr, kf, X, y, accuracy_score) 

apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)



media_auc_val, media_auc_train = avalia_classificador(lr, kf, X, y, roc_auc_score) 

apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)
from sklearn import svm

svc = svm.SVC(gamma='auto')
media_acuracia_val, media_acuracia_train = avalia_classificador(svc, kf, X, y, accuracy_score) 

apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)



media_auc_val, media_auc_train = avalia_classificador(svc, kf, X, y, roc_auc_score) 

apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
media_acuracia_val, media_acuracia_train = avalia_classificador(neigh, kf, X, y, accuracy_score) 

apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)



media_auc_val, media_auc_train = avalia_classificador(neigh, kf, X, y, roc_auc_score) 

apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)
from sklearn import tree

dt = tree.DecisionTreeClassifier(max_depth=3)
media_acuracia_val, media_acuracia_train = avalia_classificador(dt, kf, X, y, accuracy_score) 

apresenta_metrica('Acurácia', media_acuracia_val, media_acuracia_train, percentual=True)



media_auc_val, media_auc_train = avalia_classificador(dt, kf, X, y, roc_auc_score) 

apresenta_metrica('AUC', media_auc_val, media_auc_train, percentual=True)
from sklearn import tree

import graphviz 

dot_data = tree.export_graphviz(dt, out_file=None, 

                                feature_names=['Age', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S', '1', '2', '3', 'female', 'male'],  

                                filled=True, rounded=True,  

                                special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 