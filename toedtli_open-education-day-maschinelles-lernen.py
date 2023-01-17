#Drucke einen String. Die Ausgabe erscheint in der Ausgabezelle darunter:
print("Hello World!")
#hier verwendete Bibliotheken:
import numpy as np # Matrizen, lineare Algebra
import pandas as pd # Datenverarbeitung, CSV-Input mit pd.read_csv

# Die Inputdaten liegen unter "../input/" bereit
# Klicken Sie in diese Zelle und drücken Sie Shift-Enter, um sie zu evaluieren.

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Liste alle Dateien im Verzeichnis "../input/iris/". Das "!" bedeutet, dass dies nicht ein Python-Befehl ist, sondern an die Kommandozeile weitergegeben wird.
!ls ../input/iris
#Wie sehen die ersten Textzeilen des Datensatzes aus?
#Ausrufezeichen: Kommandozeilenbefehl in der Shell (nicht in Python)
!head ../input/iris/Iris.csv
#Importiere die .csv-Datei mit Pandas:
df = pd.read_csv('../input/iris/Iris.csv',index_col='Id')
df.head(7) #Zeige die ersten 7 Zeilen des Datensatzes
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']
#Importiere die Entscheidungsbaum-Klasse:
from sklearn.tree import DecisionTreeClassifier #scikit-learn
#Instanziiere einen Entscheidungsbaum:
isDecisionTree=True
clf = DecisionTreeClassifier()

from sklearn.svm import SVC
clf = SVC(C=1.0,kernel='rbf',gamma=0.1)
isDecisionTree=False 
clf.fit(X,y)
#Lösche ev. vorhandene frühere Versionen des Outputs:
if isDecisionTree:
    from sklearn import tree
    !rm -f output.png
    tree.export_graphviz(clf,out_file='entscheidungsbaum.dot',
                         feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
                         class_names = clf.classes_)
    !dot -Tpng entscheidungsbaum.dot > output.png
neue_blume_X = [4.9,3.4,1.6,0.2]
clf.predict([neue_blume_X])
from sklearn.metrics import accuracy_score
yhat = clf.predict(X)
acc = accuracy_score(yhat,y)
print(f'Trainingsgenauigkeit: {acc:1.3f}')
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,train_size=0.8,test_size=0.2)
Xtrain.shape,ytrain.shape,Xtest.shape,ytest.shape
Entscheidungsbaum.fit(Xtrain,ytrain)

yhat_test = clf.predict(Xtest)

acc_test = accuracy_score(yhat_test,ytest)
print(f"Testgenauigkeit:{acc:1.3f}")
!ls ../input
!ls ../input/titanic
#Wir laden den Trainings- und den Testdatensatz:
train = pd.read_csv('../input/titanic/train.csv',index_col='PassengerId')
test = pd.read_csv('../input/titanic/test.csv',index_col='PassengerId')
train.shape,test.shape
train.head()
test.head()
#Codiere die Wörter 'female' und 'male' als 0 und 1:
train['Sex'] = train['Sex'].map({'female':0,'male':1})
test['Sex'] = test['Sex'].map({'female':0,'male':1})
#Ersetze alle NaN ("Not-a-Number")-Werte durch Null
train_cols = ['Pclass','Fare','Sex','Age']
Xtrain = train[train_cols].fillna(0) #Noch mehr Feature-Engineering, das wir hier übergehen
Xtest = test[train_cols].fillna(0)
Xtrain.nunique()
zielspalte = 'Survived'
ytrain = train[zielspalte]
#Importiere die Entscheidungsbaum-Klasse:
from sklearn.tree import DecisionTreeClassifier 
#Instanziiere einen Entscheidungsbaum:
Entscheidungsbaum = DecisionTreeClassifier()
Entscheidungsbaum.fit(Xtrain,ytrain)
y_Vorhersage = Entscheidungsbaum.predict(Xtest)
!head ../input/titanic/gender_submission.csv
my_submission = pd.DataFrame({'PassengerId':Xtest.index,'Survived':y_Vorhersage})
my_submission.head()
# Der Dateiname ist beliebig
my_submission.to_csv('submission.csv', index=False)
from math import sin
import numpy as np
import matplotlib.pyplot as plt
def f(x):
    y = sin(1/(x-4))
    return y

xx = np.linspace(1,4.995,100)
y = [f(x) for x in xx]
plt.plot(xx,y,'x');