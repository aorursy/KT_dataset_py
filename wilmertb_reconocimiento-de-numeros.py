import numpy as np # algebra lineal
import pandas as pd # procesamiento de datos
import matplotlib.pyplot as plt # hacer graficos
# importa la base de datos de los digitos
from sklearn.datasets import load_digits
digits = load_digits()

# averigua que contiene la base digits
print("shape: ", digits.data.shape)
print("keys: ", digits.keys())
print("target names: ", digits['target_names'])
# visualización de las imagenes
images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5))
for index, (image, label) in enumerate(images_and_labels[:20]):
    plt.subplot(2, 10, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)
# se separa los datos con train_test_split
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(digits['data'],digits['target'])
# se trabaja con metodo bayesiano
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(xtrain, ytrain)
from sklearn import metrics
print('la exactitud por el metodo bayesiano es:')
print(metrics.accuracy_score(ytest, nb.predict(xtest)))
print('la exactitud para los datos de entrenamiento es:')
print(metrics.accuracy_score(ytrain, nb.predict(xtrain)))
# se trabaja con arbol de decision
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=5)
dtc.fit(xtrain, ytrain)
print('la exactitud por el metodo de arbol de decisión es:')
print('por la funcion de metricas es:', metrics.accuracy_score(ytest, dtc.predict(xtest)))
print('por la funcion score es:',dtc.score(xtest, ytest))
print('la exactitud para los datos de entrenamiento es:',dtc.score(xtrain, ytrain))
# para dibujar el arbol de decisión
from sklearn.tree import export_graphviz
export_graphviz(dtc, out_file = 'dtc.dot')
import graphviz
with open('dtc.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
# se trabaja con vecinos mas cercanos K-NN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)
print('la exactitud por el metodo K-NN es:')
print('por la funcion de metricas es:',metrics.accuracy_score(ytest, knn.predict(xtest)))
print('por la funcion score es:',knn.score(xtest, ytest))
print('la exactitud para los datos de entrenamiento es:',knn.score(xtrain, ytrain))
from sklearn.decomposition import PCA
import sklearn

#pca = PCA(n_components=2)
pca = PCA()
pca.fit(digits.data)
digits_pca = pca.transform(digits.data)

plt.scatter(digits_pca[:, 0], digits_pca[:, 1], digits.target)
plt.legend(digits.target_names, loc='best')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

print("shape de la base de datos sin tratar: ", digits.data.shape)
print("shape de la base de datos con PCA: ", digits_pca.shape)
from sklearn.preprocessing import MinMaxScaler
escala=MinMaxScaler()
escala.fit(digits.data)
escalada = escala.transform(digits.data)
pca.fit(escalada)
digits_pre = pca.transform(escalada)

plt.scatter(digits_pre[:, 0], digits_pre[:, 1], digits.target)
plt.legend(digits.target_names, loc='best')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

print("shape de la base de datos sin tratar: ", digits.data.shape)
print("shape de la base de datos con PCA: ", digits_pca.shape)
# se separa los datos con train_test_split para información con PCA
from sklearn.model_selection import train_test_split
xpcatrain, xpcatest, ypcatrain, ypcatest = train_test_split(digits_pca,digits['target'])

# se separa los datos con train_test_split para información con PCA
from sklearn.model_selection import train_test_split
xpretrain, xpretest, ypretrain, ypretest = train_test_split(digits_pre,digits['target'])
# se trabaja con metodo bayesiano
from sklearn.naive_bayes import GaussianNB
nbpca = GaussianNB()
nbpca.fit(xpcatrain, ypcatrain)
from sklearn import metrics
print('la exactitud por el metodo bayesiano es:')
print(metrics.accuracy_score(ypcatest, nbpca.predict(xpcatest)))
print('la exactitud para los datos de entrenamiento es:')
print(metrics.accuracy_score(ypcatrain, nbpca.predict(xpcatrain)))
# se trabaja con metodo bayesiano
from sklearn.naive_bayes import GaussianNB
nbpre = GaussianNB()
nbpre.fit(xpretrain, ypretrain)
from sklearn import metrics
print('la exactitud por el metodo bayesiano es:')
print(metrics.accuracy_score(ypretest, nbpre.predict(xpretest)))
print('la exactitud para los datos de entrenamiento es:')
print(metrics.accuracy_score(ypretrain, nbpre.predict(xpretrain)))
# se trabaja con arbol de decision
from sklearn.tree import DecisionTreeClassifier
dtcpca = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=5)
dtcpca.fit(xpcatrain, ypcatrain)
print('la exactitud por el metodo de arbol de decisión es:')
print('por la funcion de metricas es:', metrics.accuracy_score(ypcatest, dtcpca.predict(xpcatest)))
print('por la funcion score es:',dtcpca.score(xpcatest, ypcatest))
print('la exactitud para los datos de entrenamiento es:',dtcpca.score(xpcatrain, ypcatrain))
# se trabaja con arbol de decision
from sklearn.tree import DecisionTreeClassifier
dtcpre = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=5)
dtcpre.fit(xpretrain, ypretrain)
print('la exactitud por el metodo de arbol de decisión es:')
print('por la funcion de metricas es:', metrics.accuracy_score(ypretest, dtcpre.predict(xpretest)))
print('por la funcion score es:',dtcpre.score(xpretest, ypretest))
print('la exactitud para los datos de entrenamiento es:',dtcpre.score(xpretrain, ypretrain))
# se trabaja con vecinos mas cercanos K-NN
from sklearn.neighbors import KNeighborsClassifier
knnpca = KNeighborsClassifier(n_neighbors=3)
knnpca.fit(xpcatrain, ypcatrain)
print('la exactitud por el metodo K-NN es:')
print('por la funcion de metricas es:',metrics.accuracy_score(ypcatest, knnpca.predict(xpcatest)))
print('por la funcion score es:',knnpca.score(xpcatest, ypcatest))
print('la exactitud para los datos de entrenamiento es:',knnpca.score(xpcatrain, ypcatrain))
# se trabaja con vecinos mas cercanos K-NN
from sklearn.neighbors import KNeighborsClassifier
knnpre = KNeighborsClassifier(n_neighbors=3)
knnpre.fit(xpretrain, ypretrain)
print('la exactitud por el metodo K-NN es:')
print('por la funcion de metricas es:',metrics.accuracy_score(ypretest, knnpre.predict(xpretest)))
print('por la funcion score es:',knnpre.score(xpretest, ypretest))
print('la exactitud para los datos de entrenamiento es:',knnpre.score(xpretrain, ypretrain))