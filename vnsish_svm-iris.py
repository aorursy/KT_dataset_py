import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn import svm
#Iris dataset
data = pd.read_csv('../input/iris.data.csv');

#Dividindo em 2 classes e selecionando 2 atributos 
data = data[:-50]
x = data.iloc[:, 0:2]
#x = data.iloc[:,np.r_[0,3]]
y = data.iloc[:,-1]

#Dividindo os dados em treino e teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

pd.concat([x_train, y_train], axis=1, sort=False).head()

#Calculo do modelo
clf = svm.SVC(kernel='linear', C=100)
clf.fit(x_train, y_train)

#Plota os pontos
colors = {'Iris-setosa' : 'b',
          'Iris-versicolor' : 'r',
          'Iris-virginica' : 'g'}

c = [colors[val] for val in y_train]

plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=c, s=30)

from matplotlib.lines import Line2D
legend = [Line2D([0], [0], color='w', markerfacecolor='b', marker='o', label='Iris-setosa', markersize=7),
          Line2D([0], [0], color='w', markerfacecolor='r', marker='o', label='Iris-versicolor', markersize=7),
          Line2D([0], [0], color='w', markerfacecolor='g', marker='o', label='Iris-virginica', markersize=7)]

plt.legend(handles=legend, loc=1)
# Obter e plotar a fronteira de decisão
plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=c, s=30)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.legend(handles=legend, loc=1)
# Teste do modelo
y_pred = clf.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
