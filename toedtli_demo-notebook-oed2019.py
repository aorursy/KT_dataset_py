import numpy as np

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
!ls ../input
df_train = pd.read_csv('../input/train.csv',index_col='Id')

df_test = pd.read_csv('../input/testX.csv',index_col='Id')

df_train.head()
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(df_train[['x1','x2']],df_train.y)
vorhersage = clf.predict(df_test)
df_out = pd.Series(vorhersage,name='y')

df_out.index.name='Id'

df_out.to_csv('submission.csv',header=True)
def visualize_class_boundaries(Xtrain,ytrain,k=20,figsize=(5,5),fig=None,title=''):

    """

    Diese Funktion visualisiert die Vorhersagegrenzen für einen k-NN-Klassifikator auf 

    diesem Datensatz. Es wird ein k-NN-Klassifikator auf den Daten Xtrain, ytrain trainiert und anschliessend 

    die Trainingsdatenpunkte sowie die Klassengrenzen visualisiert. Voraussetzung ist, dass der Datensatz 

    2-dimensional ist, d.h. Xtrain.shape[1] muss den Wert 2 haben.

    """

    #Generiere das Gitter: Alle Datenpunkte {(x,y) für 0<x,y<1}

    h = .002  # Maschenabstand

    x_min = 0

    y_min = 0

    x_max = 1

    y_max = 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                         np.arange(y_min, y_max, h))

    

    if not fig:

        fig = plt.figure(1,figsize=figsize)

    clf = KNeighborsClassifier(n_neighbors=k)

    clf.fit(Xtrain,ytrain);

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z,cmap='bwr');

    plt.contourf(xx, yy, Z,alpha=0.1,cmap='bwr');

    plt.plot(Xtrain[ytrain==0,0],Xtrain[ytrain==0,1],

             marker='.',ls='none',c='b');

    plt.plot(Xtrain[ytrain==1,0],Xtrain[ytrain==1,1],

             marker='+',ls='none',c='r');

    plt.xlabel('x')

    plt.ylabel('y');

    if not title:

        plt.title(f'k={k} nächste Nachbarn')

    else:

        plt.title(title)
Xtrain = df_train[['x1','x2']].values

ytrain=df_train['y'].values

#Visualisiere die Klassengrenzen auf den Trainingsdaten:

visualize_class_boundaries(Xtrain,ytrain,k=2)  # <-- HIER DEN WERT VON k (durch Ausprobieren) OPTIMIEREN
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

clf = KNeighborsClassifier()

gs = GridSearchCV(clf,{'n_neighbors':[1,2,3,4,5,6,7,8,9,10,20,50]},#die möglichen Werte die durchprobiert werden

                  cv=10, #Teile den Datensatz (10 mal) in 10 Subsets. Trainiere auf 9, validiere auf dem letzen.

                  error_score=np.nan #Sollte ein fit fehlschlagen, bewerte Ihn mit Nan. Dies unterdrückt lästige eine Warnung.

                 )
#In Scikit-Learn lassen sich leicht andere Klassifikatoren ausprobieren. Für einen Entscheidungsbaum einfach 

#die folgenden zwei Zeilen entkommentieren und (als Übung) die Fehlermeldungen korrigieren, die entstehen, weil

#der Hyperparameter nun nicht mehr `n_neighbors`, sondern `max_depth` heisst. 

#clf = DecisionTreeClassifier()

#gs = GridSearchCV(clf,{'max_depth':[1,2,3,4,5,6,7,20]},cv=10,error_score=np.nan)
gs.fit(df_train[['x1','x2']],df_train.y)
df = pd.DataFrame(gs.cv_results_)

df.head()
ax = df.plot(x='param_n_neighbors',y='mean_train_score')

df.plot(x='param_n_neighbors',y='mean_test_score',ax=ax,title='Parameterkurven für kNN',xlim=[0,20]);
#Extrahieren den besten im Grid-Search gefundenen Wert für den Hyperparameter n_neighbours

k_best = gs.best_params_['n_neighbors']

k_best
visualize_class_boundaries(Xtrain,ytrain,k=k_best)  # <-- HIER DEN WERT VON k (durch Ausprobieren) OPTIMIEREN
#Entkommentieren für Entscheidungsbaum: (Lösung der Übung):

"""

clf = DecisionTreeClassifier()

gs = GridSearchCV(clf,{'max_depth':[1,2,3,4,5,6,7,20]},cv=10,error_score=np.nan)

gs.fit(df_train[['x1','x2']],df_train.y)

df = pd.DataFrame(gs.cv_results_)

fig,axes = plt.subplots(1,2,figsize=(15,5))

df.plot(x='param_max_depth',y='mean_train_score',title='Parameterkurven für DecisionTree',xlim=[0,20],ax=axes[0])

df.plot(x='param_max_depth',y='mean_test_score',title='Parameterkurven für DecisionTree',xlim=[0,20],ax=axes[0]);

max_depth_best =gs.best_params_['max_depth']

visualize_class_boundaries(Xtrain,ytrain,k=max_depth_best,title=f'DecisionTree (max_depth={max_depth_best})')  

""";
clf = gs.best_estimator_
yhat = clf.predict(df_test)

submission = pd.Series(yhat,name='y')

submission.index.name='Id'

submission.to_csv('submission.csv',header=True)

submission.head()
!head submission.csv