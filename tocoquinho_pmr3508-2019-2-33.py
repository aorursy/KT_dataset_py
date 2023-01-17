import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
data = {}

labels = [

    "Age","Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status","Occupation", "Relationship", 

    "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country", "Target"

]



data["train"] = pd.read_csv('../input/adult-dataset/adult.data',

        names=labels,

        sep=r'\s*,\s*',

        engine='python',

        na_values = "?")



data["test"] = pd.read_csv('../input/adult-dataset/adult.test',

        names=labels,

        engine='python',

        na_values = "?")
#Remoção dos valores ausentes

data["train"] = data["train"].dropna()

data["test"] = data["test"].dropna()



#Transformação dos dados em valores numéricos

data["train"] = data["train"].apply(preprocessing.LabelEncoder().fit_transform)

data["test"] = data["test"].apply(preprocessing.LabelEncoder().fit_transform)



data["train"].head()
corr_matrix = data["train"].corr();



plt.style.use("classic")

fig = plt.figure(facecolor="white")

ax = fig.add_subplot(111)

cax = ax.matshow(corr_matrix, vmin = -1, vmax = 1)

fig.colorbar(cax)

ticks = np.arange(0,len(labels),1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(labels)

ax.set_yticklabels(labels)

plt.setp(ax.get_xticklabels(), rotation=60)

ax.set_xlabel("Correlation matrix of the Adult Dataset")



for i in range(len(labels)):

    for j in range(len(labels)):

        try:

            text = ax.text(j, i, np.round(corr_matrix[i,j], decimals = 2), ha = "center", va="center", color="white")

        except:

            pass

            

fig.tight_layout()

plt.show()
Xtrain = data["train"]

Ytrain = data["train"].Target



Xtest = data["test"]

Ytest = data["test"].Target



removeLabels = ["Workclass", "fnlwgt", "Occupation", "Country", "Education", "Target"]

for rLabel in removeLabels:

    del Xtrain[rLabel]

    del Xtest[rLabel]
start = time.time()



RFClassifier = RandomForestClassifier(n_estimators = 100, max_depth = 8, warm_start = True)



RFCV = cross_val_score(RFClassifier, Xtrain, Ytrain, cv = 10)



RFClassifier.fit(Xtrain,Ytrain)



end = time.time()
Yprev = RFClassifier.predict(Xtest)

print("Acurácia Random Forest: {}".format(accuracy_score(Ytest,Yprev)))



print("Acurácia da validação cruzada da RF: {}".format(np.mean(RFCV)))



print("Tempo decorrido:", end - start, "segundos")

start = time.time()



LRClassifier = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter = 1000, warm_start = False)



LRCV = cross_val_score(LRClassifier, Xtrain, Ytrain, cv=10)



LRClassifier.fit(Xtrain, Ytrain)



end = time.time()
Yprev = LRClassifier.predict(Xtest)

print("Acurácia Regressão Logística: {}".format(accuracy_score(Ytest,Yprev)))



print("Acurácia da validação cruzada da RL: {}".format(np.mean(LRCV)))



print("Tempo decorrido:", end - start, "segundos")
start = time.time()



NNClassifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,5), random_state=1)



NNCV = cross_val_score(NNClassifier, Xtrain, Ytrain, cv=10)



NNClassifier.fit(Xtrain, Ytrain)

    

end = time.time()
Yprev = NNClassifier.predict(Xtest)

print("Acurácia Rede Neural: {}".format(accuracy_score(Ytest,Yprev)))



print("Acurácia da validação cruzada da RN: {}".format(np.mean(NNCV)))



print("Tempo decorrido:", end - start, "segundos")