import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# The Iris Setosa

from IPython.display import Image

url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'

Image(url,width=300, height=300)
# The Iris Versicolor

from IPython.display import Image

url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'

Image(url,width=300, height=300)
# The Iris Virginica

from IPython.display import Image

url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'

Image(url,width=300, height=300)
iris = sns.load_dataset('iris')

iris.head()
# Normal pairplot with numerical data

sns.pairplot(iris)
# sepating the data as per the species

sns.pairplot(iris, hue= 'species')
sns.set_style('darkgrid')

setosa = iris[iris['species']=='setosa']

sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)
versicolor = iris[iris['species']=='versicolor']

sns.kdeplot(versicolor['sepal_width'], versicolor['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)
virginica = iris[iris['species']=='virginica']

sns.kdeplot(virginica['sepal_width'], virginica['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split



X = iris.drop('species', axis=1)

y = iris['species']



train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=100)



model = SVC()

model.fit(train_X, train_y)



prediction = model.predict(test_X)
print(confusion_matrix(test_y, prediction))

print('\n')

print(classification_report(test_y, prediction))