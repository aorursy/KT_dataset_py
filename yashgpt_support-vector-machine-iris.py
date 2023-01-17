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
import seaborn as sns

iris=sns.load_dataset('iris')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib notebook
%matplotlib notebook

sns.set_style('whitegrid')

sns.pairplot(iris,palette='Dark2',hue='species')
%matplotlib inline

setosa=iris[iris['species']=='setosa']

sns.kdeplot(setosa['sepal_width'],setosa['sepal_length'],cmap='plasma',shade=True,shade_lowest=False)
from sklearn.model_selection import train_test_split
X=iris.drop('species',axis=1)

y=iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
from sklearn.svm import SVC
svc_model=SVC(gamma='auto')
svc_model.fit(X_train,y_train)
pred=svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}
grid=GridSearchCV(SVC(),param_grid,verbose=2)

grid.fit(X_train,y_train)
g_pred=grid.predict(X_test)

print(grid.best_estimator_)

print(grid.best_params_)
print(confusion_matrix(y_test,g_pred))
print(classification_report(y_test,g_pred))