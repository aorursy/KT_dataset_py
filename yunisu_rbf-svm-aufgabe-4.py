
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

import numpy as np
import pandas as pd 

import os  
print(os.listdir("../input"))

#Daten der CSV Datei train werden eingelesen
train = pd.read_csv("../input/train.csv")

#Import von Support Vector Machine
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


data = np.array(train)
#Alle Werte der Spalten X & Y
X = data[:,:2]
#Nur die Werte der Spalte class
Y = data [:,2]

#Test der Daten
#print(Y.shape)
#print(X)

#Training und Test Samples erstellen
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)  

#Radial basis function SVM implementieren, da die Accuracy der Daten mit Linear SVC nur bei 50% liegen würde
model =svm.SVC(kernel='rbf',gamma='scale', C=1E6)

#Training der Daten
model.fit(X_train,Y_train)

#Accuracy/ Evaluation der Samples
score1 = model.score(X_train,Y_train)
score2 = model.score(X_test,Y_test)
print(score1)
print(score2)

#Visualisierung
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_decision_boundary(model,X,y):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min(), X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min(), X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
              edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(model.__class__.__name__)

    plt.show()

plot_decision_boundary(model,X_train,Y_train)




