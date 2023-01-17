#wir brauchen unsere Standardbibliotheken

import numpy as np #für vektoren und Matrizen

import matplotlib.pyplot as plt #für plots und charts

#zeichne Grafiken direkt ins notebook:

%matplotlib inline 
from sklearn.datasets import make_moons

Xtrain,ytrain = make_moons(n_samples=100000, noise=.4)

Xvalid,yvalid = make_moons(n_samples=100000, noise=.4)



#Schauen wir uns die Trainingsdaten an (NICHT die Testdaten!)

plt.scatter(Xtrain[ytrain==0,0],Xtrain[ytrain==0,1],c='r');

plt.scatter(Xtrain[ytrain==1,0],Xtrain[ytrain==1,1],c='b');

plt.xlabel('Feature 1')

plt.ylabel('Feature 2')

plt.title('Unser Trainingsdatensatz');
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)

clf.fit(Xtrain,ytrain)



clf.score(Xtrain,ytrain)#Juhuu!...? 100% Genauigkeit!?
clf.score(Xvalid,yvalid)
#clf.score?
from sklearn.metrics import accuracy_score, confusion_matrix

clf.fit(Xtrain,ytrain)

yhat_valid = clf.predict(Xvalid)

print(confusion_matrix(yvalid,yhat_valid)) #besonders nützlich bei mehr als 2 Klassen!

train_acc = accuracy_score(yvalid,yhat_valid)

print("Trainingsgenauigkeit: {0:3.0f}%".format(100*train_acc))
from sklearn.metrics import classification_report

print(classification_report(yvalid,yhat_valid))
from sklearn.datasets import make_moons

#make_moons? 

X,y = make_moons(n_samples=1000,noise=0.8)

from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(max_depth=5)
from sklearn.model_selection import learning_curve

#Entkommentieren Sie die folgende Zeile! Lernen Sie, was diese Funktion genau macht.

#learning_curve?
train_sizes=np.linspace(0.001,1,20)

train_sizes, train_scores, test_scores = learning_curve(

        clf, X, y, cv=10, train_sizes=train_sizes)
train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

plt.grid()



plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                 train_scores_mean + train_scores_std, alpha=0.1,

                 color="r")

plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

         label="Training score")

plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

         label="Cross-validation score")

plt.xlabel('Anzahl Trainingszeilen')

plt.ylabel('Genauigkeit')

plt.title('Lernkurven\nauf einem Moons-Datensatz')

plt.legend();
import sklearn.datasets

iris=sklearn.datasets.load_iris()

X = iris['data']

y = iris['target'] 

#iris enthält noch weitere keys: welche?
from sklearn.model_selection import train_test_split

#Entkommentieren Sie die folgende Zeile. 

#train_test_split?
Xtrain,Xvalid,ytrain,yvalid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)

Xtrain.shape,Xvalid.shape,ytrain.shape,yvalid.shape
#ein 1-NN Klassifikator...

kNN1 = KNeighborsClassifier(n_neighbors=1)

#... und ein 50-NN Klassifikator:

kNN50 = KNeighborsClassifier(n_neighbors=50)
kNN50.fit(Xtrain,ytrain)

score50=kNN50.score(Xtrain,ytrain)

kNN1.fit(Xtrain,ytrain)

score1=kNN1.score(Xtrain,ytrain)

#print(score50,score1) # Entkommentieren Sie diese Zeile, sobald Sie die obige Übung erledigt haben.

#Python-Übung:Erstellen Sie einen String, der eine "Sätzliantwort" gibt. "Die Genauigkeit beträgt..."
"""

kNN1.fit(,) #hier soll eine X- und eine y-Matrix übergeben werden

kNN1.score(,)



kNN50.fit(,)

kNN50.score(,)

"""
Xtrain,Xvalid,ytrain,yvalid = train_test_split(X,y,train_size=0.8,test_size=0.2)

#Besser: 

#Xtrain,Xvalid,ytrain,yvalid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf,Xtrain,ytrain,cv=10)

np.around(scores,2)
#Bitte entkommentieren und verbessern Sie:

#mittlere_Genauigkeit = Ihre_Funktion(scores)

#standardfehler = noch_eine_Funktion(scores)

#print("Trainingsgenauigkeit: {0:3.0f}% +/-{1:2.0f}%".format(100*mittlere_Genauigkeit,100*standardfehler))
from sklearn.model_selection import validation_curve

#Entkommentieren Sie die folgende Zeile! Lernen Sie, was diese Funktion genau macht.

#validation_curve?
X,y = make_moons(n_samples=1000,noise=0.8)

Xtrain,Xvalid,ytrain,yvalid = train_test_split(X,y,train_size=0.8,test_size=0.2)



Xtrain.shape
kNN1 = KNeighborsClassifier(n_neighbors=5)



param_values=np.arange(1,500,30)

train_scores,test_scores = validation_curve(kNN1,Xtrain[:1000,:],ytrain[:1000],

                 'n_neighbors',param_values,cv=10)

train_scores.shape,test_scores.shape,param_values.shape
plt.plot(param_values,np.mean(train_scores,axis=1),label='train');

plt.plot(param_values,np.mean(test_scores,axis=1),label='test');

plt.xlabel('$k$'),

plt.ylabel('Genauigkeit')

plt.legend();
train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

train_scores.shape,train_scores_mean.shape
np.mean(y)


plt.grid()

plt.fill_between(param_values, train_scores_mean - train_scores_std,

                 train_scores_mean + train_scores_std, alpha=0.1,

                 color="r")

plt.fill_between(param_values, test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(param_values, train_scores_mean, 'o-', color="r",

         label="Training score")

plt.plot(param_values, test_scores_mean, 'o-', color="g",

         label="Cross-validation score")

plt.xlabel('k')

plt.ylabel('Genauigkeit')

plt.title('Validierungskurven\nauf einem Moons-Datensatz')

plt.legend();
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()

Xtrain,Xvalid,ytrain,yvalid = train_test_split(bc['data'],bc['target'])

Xtrain.shape,ytrain.shape,Xvalid.shape,yvalid.shape
from sklearn.tree import DecisionTreeClassifier

dt1 = DecisionTreeClassifier(max_depth=2)



param_values=np.arange(1,25,2)

train_scores,test_scores = validation_curve(dt1,Xtrain,ytrain,

                 'max_depth',param_values,cv=10)
plt.plot(param_values,np.mean(train_scores,axis=1),label='train');

plt.plot(param_values,np.mean(test_scores,axis=1),label='test');

plt.xlabel('max_depth'), plt.ylabel('Genauigkeit')

plt.title('Parameterkurve für max_depth eines Entscheidungsbaums \nauf dem Wisconsin Breast Cancer Datensatz')

plt.ylim(0.8,1.05)

plt.legend();
from sklearn import datasets

X, y = datasets.make_classification(n_samples=10000, n_features=20,

                                    n_informative=2, n_redundant=10,

                                    random_state=42)
from sklearn.model_selection import validation_curve

param_range=np.arange(1,40,5)

clf= DecisionTreeClassifier()

train_scores, test_scores = validation_curve(clf

    , X, y, param_name="max_depth", param_range=param_range,

    cv=5, scoring="accuracy", n_jobs=1)

train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

plt.ylim(0.8,1.05)

plt.title("Validation Curve")

plt.xlabel(r"max_depth")

plt.ylabel("Genauigkeit")

lw = 2

plt.plot(param_range, train_scores_mean, label="Training score",

             color="darkorange", lw=lw) #oder plt.semilogx

plt.fill_between(param_range, train_scores_mean - train_scores_std,

                 train_scores_mean + train_scores_std, alpha=0.2,

                 color="darkorange", lw=lw)

plt.plot(param_range, test_scores_mean, label="Cross-validation score",

             color="navy", lw=lw) #oder plt.semilogx

plt.fill_between(param_range, test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std, alpha=0.2,

                 color="navy", lw=lw)

plt.legend(loc="best")



plt.show()