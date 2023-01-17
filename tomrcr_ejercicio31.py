import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import sklearn.preprocessing
import sklearn.model_selection

import sklearn.cluster
import umap
import sklearn.neural_network
from sklearn.metrics import plot_confusion_matrix
#Importa y adecúa los datos

dataCases = pd.read_csv('../input/uncover/UNCOVER/our_world_in_data/coronavirus-disease-covid-19-statistics-and-research.csv')
dataMobility = pd.read_csv('../input/uncover/UNCOVER/google_mobility/regional-mobility.csv')

dataMobility = dataMobility.sort_values(by=['country','date'])
parciales = dataMobility[dataMobility['region']!='Total'].index
dataMobility = dataMobility.drop(parciales,axis=0).drop(['region'],axis=1)
dataMobility = dataMobility.fillna(0)

dataCases = dataCases.rename(columns = {'location':'country'})
fechasSinCasos = dataCases[dataCases['total_cases']==0].index
dataCases = dataCases.drop(fechasSinCasos,axis=0) # Sólo considera fechas después del primer caso
dataCases = dataCases.sort_values(by=['country','date'])
dataCases = dataCases.drop(['iso_code','total_tests','new_tests','total_tests_per_thousand', 'new_tests_per_thousand', 'tests_units'],axis=1)
dataCases['date'] = pd.to_datetime(dataCases['date'])
# Omite los países que no se encuentren en ambos datasets

paisesC = dataCases['country'].unique()
paisesM = dataMobility['country'].unique()
paises = []

for pais in paisesM:
    a = dataMobility[dataMobility['country']==pais]
    b = a.shape
    if b[0] < 63: # Remuevo los países que no han reportado alguna fecha
        dataMobility = dataMobility.drop(a.index,axis=0)
        #print('*'+pais)
    elif pais not in paisesC: #Remuevo los países que no reportaron casos
        dataMobility = dataMobility.drop(a.index,axis=0)
        #print('.'+pais)
    else:
        paises.append(pais)

for pais in paisesC:
    a = dataCases[dataCases['country']==pais]
    if pais not in paises: #Remuevo los paises que no reportaron movilidad
        dataCases = dataCases.drop(a.index,axis=0)
        #print(pais)
# Toma los datos que se usarán

movPaises = []
for pais in paises:
    a = dataMobility[dataMobility['country']==pais]
    datosPais = np.array(a.drop(['country','date'],axis=1))
    movPaises.append(datosPais)

tasaCasos = [] # Tasa promedio (diaria) de casos, por cada millón de habitantes
for pais in paises:
    a = dataCases[dataCases['country']==pais]
    fechas = np.array(a['date'],dtype='datetime64[D]')
    propCasos = np.array(a['total_cases_per_million'])[-1]
    dias = (fechas[-1]-fechas[0]).astype(int)
    tasaCasos.append(propCasos/dias)
# Reescala y reformatea los datos

movPaises = np.array(movPaises)
X = movPaises.reshape(len(movPaises),-1)
Y = np.array(tasaCasos)

scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X)
plt.figure(figsize=(35,10))
plt.plot(paises,Y)
plt.xticks(paises,rotation=90)
plt.ylabel('Tasa de casos')
plt.grid()
plt.title('Casos nuevos, por día, por millón de habitantes de cada país')
plt.show()
# Visualización reducida por UMAP con clusters por k-means

reducer = umap.UMAP(n_neighbors=15)
reducer.fit(X_train)
embedding = reducer.transform(X_train)

n_clusters = 3
k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
k_means.fit(embedding)
cluster = k_means.predict(embedding)
distance = k_means.transform(embedding)
plt.figure(figsize=(15,15))
plt.scatter(embedding[:,0], embedding[:,1], cmap='Paired', s=20.0,c=cluster)
plt.title('Reducción por UMAP de los comportamientos de movilidad')

suramerica = ['Argentina','Bolivia','Brazil','Chile','Colombia','Ecuador','Panama','Paraguay','Peru','Uruguay','Venezuela']
for i in range(len(paises)):   
    x = embedding[:,0][i]
    y = embedding[:,1][i]
    color = 'steelblue'
    size = 'x-small'
    if paises[i] in suramerica:
        color = 'red'
        size = 'small'
    plt.text(x+0.01,y+0.01,paises[i],color=color,fontsize=size)
# Clasificación según región

restoLatam = ['Puerto Rico','Mexico','Nicaragua','Guatemala','Honduras','Costa Rica','El Salvador','Dominican Republic','Haiti','Aruba']
Latam = suramerica + restoLatam

Y2 = [1 if i in Latam else 0 for i in paises]
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y2, test_size=0.5)
scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = sklearn.neural_network.MLPClassifier(activation='relu',hidden_layer_sizes=(63,6),max_iter=10000)
mlp.fit(X_train, Y_train)

Loss = mlp.loss_
F1 =  sklearn.metrics.f1_score(Y_test, mlp.predict(X_test),average='macro')
m = sklearn.metrics.confusion_matrix(Y_test, mlp.predict(X_test))

plot_confusion_matrix(mlp, X_test, Y_test)
plt.title('F1 = {:.2f} ; Loss = {:.2f}'.format(F1,Loss))
plt.xticks([0,1],['Mundo','Latam'])
plt.yticks([1,0],['Latam','Mundo'])
plt.show()
Y3 = 1*(Y>np.sort(Y)[int(len(Y)/2)])

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y3, test_size=0.7)
scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = sklearn.neural_network.MLPClassifier(activation='logistic',hidden_layer_sizes=(7,9,6),max_iter=10000)
mlp.fit(X_train, Y_train)

Loss = mlp.loss_
F1 =  sklearn.metrics.f1_score(Y_test, mlp.predict(X_test),average='macro')
m = sklearn.metrics.confusion_matrix(Y_test, mlp.predict(X_test))

plot_confusion_matrix(mlp, X_test, Y_test)
plt.title('F1 = {:.2f} ; Loss = {:.2f}'.format(F1,Loss))
plt.xticks([0,1],['Bajo','Alto'])
plt.yticks([1,0],['Alto','Bajo'])
plt.show()
Y_lat = [Y[i] for i in range(len(paises)) if paises[i] in Latam ]
Y_lat = 1*(Y_lat>np.sort(Y)[int(len(Y)/2)])
X_lat = [X[i] for i in range(len(paises)) if paises[i] in Latam ]

Loss = mlp.loss_
F1 =  sklearn.metrics.f1_score(Y_lat, mlp.predict(X_lat),average='macro')
m = sklearn.metrics.confusion_matrix(Y_lat, mlp.predict(X_lat))

plot_confusion_matrix(mlp, X_lat, Y_lat)
plt.title('F1 = {:.2f} ; Loss = {:.2f}'.format(F1,Loss))
plt.xticks([0,1],['Bajo','Alto'])
plt.yticks([1,0],['Alto','Bajo'])
plt.show()
