# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/StudentsPerformance.csv')
data.info()
#Renaming Columns

data.columns = ['gender', 'race', 'parentDegree', 'lunch', 'course', 'mathScore', 'readingScore', 'writingScore']
data.isna().sum()

#No hay valores nulos
#Visualizamos los 10 primeros elementos

data.head(10)
#Calculamos mínmos y máximos para cada cosa



print('Maxima puntuación en matemáticas: ',max(data['mathScore']))

print('Mínima puntuación en matemáticas: ',min(data['mathScore']))

print('Maxima puntuación en lectura: ',max(data['readingScore']))

print('Mínima puntuación en lectura: ',min(data['readingScore']))

print('Maxima puntuación en escritura: ',max(data['writingScore']))

print('Mínima puntuación en escritura: ',min(data['writingScore']))
#Calculamos el número de estudiantes que han lagrado máximos

print('Número de estudiantes que han sacado la máxima puntación en matemáticas: ', len(data[data['mathScore'] == 100]))

print('Número de estudiantes que han sacado la máxima puntación en lectura: ', len(data[data['readingScore'] == 100]))

print('Número de estudiantes que han sacado la máxima puntación en escritura: ', len(data[data['writingScore'] == 100]))
#Estudiantes que han logrado lo máximo en las tres categorías



perfect_writing = data['writingScore'] == 100

perfect_reading = data['readingScore'] == 100

perfect_math = data['mathScore'] == 100



perfect_score = data[(perfect_math) & (perfect_reading) & (perfect_writing)]

perfect_score
print('Número de estudiantes que han sacado la máxima puntación en las tres áreas: ',len(perfect_score))
#Numero de estudiantes que ha sacado el menor valor en las tres categorías

minimum_math = data['mathScore'] == 0

minimum_reading = data['readingScore'] == 17

minimum_writing = data['writingScore'] == 10







minimum_score = data[(minimum_math) & (minimum_reading) & (minimum_writing)]

minimum_score
print('Número de estudiantes que han sacado la mínima puntación en las tres áreas: ', len(minimum_score))
#Agrupamos por raza y grado de estudios

data.groupby(['race','parentDegree']).mean()

#Analizamos la media por género

data.groupby(['gender']).mean()







#Parece que las chicas son mejores que los chicos ...
data.corr()

#Hay una fuerte correlación entre readingScore & writingScore, readingScore & mathScore and writingScore & mathScore
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.describe()
sns.set_style('darkgrid')



sns.pairplot(data, hue = 'gender')

plt.show()









color_list = ['red' if i=='female' else 'yellow' for i in data.loc[:,'gender']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'gender'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
#Procentaje de chicos y de chicas



sns.countplot(x="gender", data=data)

data['gender'].value_counts()




plt.figure(figsize=(10,4))



plt.subplot(1,3,1)

sns.barplot(x = 'gender', y = 'readingScore', data = data)



plt.subplot(1,3,2)

sns.barplot(x = 'gender', y = 'writingScore', data = data)



plt.subplot(1,3,3)

sns.barplot(x = 'gender', y = 'mathScore', data = data)



plt.tight_layout()





#Parace que a los chicos se les da mejor las matemáticas y a las chicas mejoran en lectura y en escritura
plt.figure(figsize=(14,4))



plt.subplot(1,3,1)

sns.barplot(x = 'race', y = 'readingScore', data = data)

plt.xticks(rotation = 90)



plt.subplot(1,3,2)

sns.barplot(x = 'race', y = 'writingScore', data = data)

plt.xticks(rotation = 90)



plt.subplot(1,3,3)

sns.barplot(x = 'race', y = 'mathScore', data = data)

plt.xticks(rotation = 90)



plt.tight_layout()

data.parentDegree.unique()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

x = data.drop(['race', 'parentDegree', 'lunch', 'course', 'gender'], axis=1)

y= data['gender']

knn.fit(x,y)

prediction = knn.predict(x)

print('Predicción: {}'.format(prediction))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print('La precisión de nuestro modelo es: ',knn.score(x_test,y_test))
rg = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

for i, k in enumerate(rg):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train)

    train_accuracy.append(knn.score(x_train, y_train))

    test_accuracy.append(knn.score(x_test, y_test))



plt.figure(figsize=[13,8])

plt.plot(rg, test_accuracy, label = 'Precisiñon de test')

plt.plot(rg, train_accuracy, label = 'Precisión de entrenamiento')

plt.legend()

plt.title('K VS Precisión')

plt.xlabel('K')

plt.ylabel('Precisión')

plt.xticks(rg)

plt.show()

print("La mejor precisión que podemos obtener es {} con K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
data1 = data[data['race'] =='group A']

x = np.array(data1.loc[:,'readingScore']).reshape(-1,1)

y = np.array(data1.loc[:,'writingScore']).reshape(-1,1)

# Scatter

plt.figure(figsize=[10,10])

plt.scatter(x=x,y=y)

plt.xlabel('readingScore')

plt.ylabel('writingScore')

plt.show()
from sklearn.linear_model import LinearRegression

reg = LinearRegression()



predict_space = np.linspace(min(x), max(x)).reshape(-1,1)

reg.fit(x,y)

predicted = reg.predict(predict_space)

print('R^2 score: ',reg.score(x, y))



plt.plot(predict_space, predicted, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('readingScore')

plt.ylabel('writingScore')

plt.show()
from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report



data['race_binary'] = [1 if i == 'group E' else 0 for i in data.loc[:,'race']]

x,y = data.loc[:,(data.columns != 'race') & (data.columns != 'race_binary')], data.loc[:,'race_binary']

x = x.drop(['gender', 'parentDegree', 'lunch', 'course'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

logreg = LogisticRegression()

logreg.fit(x_train,y_train)

y_pred_prob = logreg.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('Falsos positivos')

plt.ylabel('Verdaderos positivos')

plt.title('ROC')

plt.show()
from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors': np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=3)

knn_cv.fit(x,y)



# Print hyperparameter

print("Hiperparámetro K ajustado: {}".format(knn_cv.best_params_)) 

print("Mejor puntuación: {}".format(knn_cv.best_score_))
rg = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

for i, k in enumerate(rg):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train)

    train_accuracy.append(knn.score(x_train, y_train))

    test_accuracy.append(knn.score(x_test, y_test))



plt.figure(figsize=[13,8])

plt.plot(rg, test_accuracy, label = 'Precisiñon de test')

plt.plot(rg, train_accuracy, label = 'Precisión de entrenamiento')

plt.legend()

plt.title('K VS Precisión')

plt.xlabel('K')

plt.ylabel('Precisión')

plt.xticks(rg)

plt.show()

print("La mejor precisión que podemos obtener es {} con K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))