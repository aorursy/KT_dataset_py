# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Einlesen der Daten
    
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

train = read_data("../input/kiwhs-comp-1-complete/train.arff")

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = [(x,y,c) for x,y,c in train]
data = np.array(train)
#Alle numerischen Werte
X= data[:,:2]
#Nur die Ids 1 -1
Y = data [:,2]

#print(X.shape)
#print(Y)
#Training und Test Samples erstellen
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)  

#Linear SVC Implementieren
model =svm.SVC(kernel='linear', C=1.0)
model.fit(X_train,Y_train)

#Accuracy
score=model.score(X_train,Y_train)
score=model.score(X_test,Y_test)



#Visualisierung
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def plot_decision_boundary(model,X,y):
    h = .02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
              edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(model.__class__.__name__)

    plt.show()


plot_decision_boundary(model,X_train,Y_train)


pred = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")
predX = pred[["X","Y"]].values


submission = pd.DataFrame({
    'Id (String)': pd.RangeIndex(400),
    'Category (String)': model.predict(predX).astype(int)
})
submission.to_csv('submission.csv', index=False)

print(submission)

