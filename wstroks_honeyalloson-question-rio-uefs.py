import pandas as pd

import numpy as np
base = pd.read_json('/kaggle/input/tipos-de-aprendizagem-aluno/honey-allonso.json')

base_saida = pd.read_json('/kaggle/input/tipos-de-aprendizagem-aluno/label-honey-allonso.json')
base
base_saida
previsores= base.iloc[:,0:81].values

classe = base_saida.iloc[:,0].values
previsores.shape

classe.shape
classe
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#onehotencoder= OneHotEncoder(categories='auto', drop=None, sparse=True, dtype='float64', handle_unknown='error')

#previsores = onehotencoder.fit_transform(previsores).toarray()
labelenconderclasse=LabelEncoder()



classe=labelenconderclasse.fit_transform(classe)
classe
#from sklearn.preprocessing import StandardScaler

#escalona = StandardScaler()

#previsores=escalona.fit_transform(previsores)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score
def random_forest(previsores,classe):

    previsores_treinamento , previsores_teste,classe_treinamento, classe_teste=train_test_split(previsores,classe,test_size=0.25, random_state=2)

    classificador_random_florest=RandomForestClassifier(n_estimators=80,criterion='entropy',random_state=2)

    classificador_random_florest.fit(previsores_treinamento,classe_treinamento)

    previsao=classificador_random_florest.predict(previsores_teste)

    precisao=accuracy_score(classe_teste,previsao)

    matriz=confusion_matrix(classe_teste,previsao)

    return precisao, matriz
precisao,matriz=random_forest(previsores,classe)
array=[]
array.append({'modelo':"Random Florest", "Precisão":precisao})

array
precisao
matriz
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import  cross_val_score, GridSearchCV
def gbc(previsores, classe):

    previsores_treinamento , previsores_teste,classe_treinamento, classe_teste=train_test_split(previsores,classe,test_size=0.25, random_state=0) 

    np.random.seed(0)

    gbc = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1,

                                  random_state = 0)

    #cv_scores = cross_val_score(gbc, previsores_treinamento, classe_treinamento, scoring = 'roc_auc', cv = 5, n_jobs = -1)

    gbc.fit(previsores_treinamento,classe_treinamento)

    previsao=gbc.predict(previsores_teste)

    precisao=accuracy_score(classe_teste,previsao)

    matriz=confusion_matrix(classe_teste,previsao)

    return precisao,matriz;
precisao1,matriz=gbc(previsores,classe)
array.append({'modelo':"Gradient Boosting:", "Precisão":precisao1})

array
precisao1
from sklearn.tree import DecisionTreeClassifier
def tree_decision(previsores, classe):

    previsores_treinamento , previsores_teste,classe_treinamento, classe_teste=train_test_split(previsores,classe,test_size=0.25, random_state=0)

    classificador=DecisionTreeClassifier(criterion='entropy',random_state=0)

    classificador.fit(previsores_treinamento,classe_treinamento)

    previsoes=classificador.predict(previsores_teste)

    precisao=accuracy_score(classe_teste,previsoes)

    matriz=confusion_matrix(classe_teste,previsoes)

    return precisao, matriz
precisao,matriz=tree_decision(previsores,classe)
array.append({'modelo':"Tree Decision:", "Precisão":precisao})

array
precisao
import keras

from keras.models import Sequential

from keras.layers import Dense, Activation
def neural_network(previsores, classe_1):

    previsores_treinamento1, previsores_teste1,classe_treinamento1, classe_teste1=train_test_split(previsores,classe_1,test_size=0.25, random_state=0)

    classificador=Sequential()

    classificador.add(Dense(units=55,activation='relu',input_dim=80))

    classificador.add(Dense(units=55,activation='relu'))

    classificador.add(Dense(units=1,activation='sigmoid'))#camada binária retorna sigmode

    classificador.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    classificador.fit(previsores_treinamento1,classe_treinamento1,batch_size=10,epochs=100)

    previsoes1=classificador.predict(previsores_teste1)

    previsoes1=(previsoes1>0.5)

    precisao1=accuracy_score(classe_teste1,previsoes1)

    matriz1=confusion_matrix(classe_teste1,previsoes1)

    return precisao1, matriz1;
precisao1,matriz1=neural_network(previsores,classe)
precisao1
array.append({'modelo':"Neural Network:", "Precisão":precisao1})

array