import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report
# Carrega os dados

data = pd.read_csv("../input/DSL-StrongPasswordData.csv", header = 0)

# Reinicia o index

data = data.reset_index()

# Pega os IDs unicos campo 'subject'

unisub = list(data['subject'].unique())

# Cria ID numerico sequencial

mlist = [int(x) for x in range(len(unisub))]

# Vincula o Id numerico com o campo 'subject'

newvalue = dict(zip(unisub, mlist))

data['subject'] = data['subject'].map(newvalue)
# Divide o conjunto de dados utilizando a proporção 80:20

train, test = train_test_split(data, test_size = 0.2, random_state=42)



features = list(data.columns[2:])



X = train[features]

y = train['subject']



X_test = test[features]

y_test = test['subject']
#Pre processa os dados

scaler = StandardScaler()

scaler.fit(X)



scaler.transform(X)

scaler.transform(X_test)

net = MLPClassifier(random_state=42,hidden_layer_sizes=(84, ),max_iter=600,activation= 'relu', learning_rate= 'invscaling', solver='adam')
net.fit(X,y)
# predict the output using the test data on the learned model

predicted_output = net.predict(X_test)
model_accuracy = metrics.accuracy_score(y_test, predicted_output)

print('Acurácia do modelo:',model_accuracy)
print(classification_report(y_test, predicted_output, target_names=unisub))