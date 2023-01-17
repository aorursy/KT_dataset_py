#Importando bibliotecas utilizadas nesse Notebook

import pandas as pd

import sklearn

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing

from sklearn.metrics import accuracy_score
#Lendo as bases de treino e teste

adult = pd.read_csv("../input/adult-dataset/adult.test",

                    names=["Age","Workclass","fnlwgt","Education",

                           "Education-Num","Martial Status","Occupation", 

                           "Relationship","Race","Sex","Capital Gain",

                           "Capital Loss","Hours per week", "Country", "Target"],

                    engine='python',

                    sep=r'\s*,\s*',

                    na_values="?")



adultTest = pd.read_csv("../input/adult-dataset/adult.data",

                    names=["Age","Workclass","fnlwgt","Education",

                           "Education-Num","Martial Status","Occupation", 

                           "Relationship","Race","Sex","Capital Gain",

                           "Capital Loss","Hours per week", "Country", "Target"],

                    engine='python',

                    sep=r'\s*,\s*',

                    na_values="?")
#Remoção de colunas com dados faltantes das bases de dados

nadult = adult.dropna()

nadultTest = adultTest.dropna()
#Conversão das bases de teste para valores numéricos

numericAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

numericAdultTest = nadultTest.apply(preprocessing.LabelEncoder().fit_transform)
#Divisão das bases de teste em X e Y



#Vetor das labels a serem utilizadas no modelo

classes = [

    "Age", 

#     "Workclass", 

#     "fnlwgt",

#     "Education",

    "Education-Num", 

    "Martial Status",

    "Occupation", 

#     "Relationship", 

#     "Race", 

#     "Sex", 

    "Capital Gain", 

    "Capital Loss",

#     "Hours per week", 

#     "Country"

]



Xadult = numericAdult[classes]

Yadult = numericAdult.Target



XtestAdult = numericAdultTest[classes]

YtestAdult = numericAdultTest.Target
#Criação do kernel, com o número de vizinhos igual a 34, melhor valor obtido a partir de teste

knn  = KNeighborsClassifier(n_neighbors=34)

knn.fit(Xadult, Yadult)
#Predição dos valores de renda pelo knn e comparação com os valores reais.

YtestPredict = knn.predict(XtestAdult)

accuracy_score(YtestAdult, YtestPredict)
#Transform the prediction in a csv file

id_index = pd.DataFrame({'Id' : list(range(len(YtestPredict)))})

result = pd.DataFrame({'income' : YtestPredict})

result