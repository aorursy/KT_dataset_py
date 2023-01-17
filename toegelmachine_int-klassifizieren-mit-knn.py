import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# .arff-Datei (Trainingsdaten) einlesen
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() 
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

train = read_data("../input/kiwhs-comp-1-complete/train.arff")
# Panda-Dataframe aus Daten erstellen
df_data = pd.DataFrame({'x':[item[0] for item in train], 'y':[item[1] for item in train], 'Category':[item[2] for item in train]})

# Zur Überprüfung erste Zeilen des Dataframes ausgeben
df_data.head()
from sklearn.model_selection import train_test_split

# Dataframe X enthält alle x und y-Variablen für alle Punkte
X = df_data[["x","y"]].values
# Dataframe Y enthält Kategorien der Punkte
Y = df_data["Category"].values
# Farben zur Visualisierung definieren
colors = {-1:'red',1:'blue'}

# Daten aufsplitten
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=0, test_size = 0.2)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_Train)

# Gesplittete Daten skalieren, damit diese bei Visualisierung geordnet aussehen
X_Train = scaler.transform(X_Train)
X_Test = scaler.transform(X_Test)
%matplotlib inline
import matplotlib.pyplot as plt

# Trainingsdaten visualisieren und Farben zuweisen
plt.scatter(X[:,0],X[:,1],c=df_data["Category"].apply(lambda x: colors[x]))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
from sklearn.neighbors import KNeighborsClassifier

test_accuracy = []

neighbors_range = range(1,50)

# Verschiedene Modelle mit verschiedenen Parametern testen, um besten Wert zu ermitteln
for n_neighbors in neighbors_range:
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_Train, Y_Train)
    test_accuracy.append(clf.score(X_Test, Y_Test))    
    
plt.plot(neighbors_range, test_accuracy, label='Genauigkeit bei den Testdaten')
plt.ylabel('Genauigkeit')
plt.xlabel('Anzahl der Nachbarn')
plt.legend()
# Model mit Wert des besten Parameters (14) berechnen
model = KNeighborsClassifier(n_neighbors = 14)
model.fit(X_Train, Y_Train)

# Score mit n_neighbors = 14 ausgeben
print(model.score(X_Test,Y_Test))
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X, Y.astype(np.integer), clf=model, legend=2, colors=('#ff4c4c,#32c8c8,#00ff00'))
# Versuch des Vorhersagens
testdf = pd.read_csv("../input/kiwhs-comp-1-complete/test.csv")

testX = testdf[["X","Y"]].values
model.predict(testX)

# Ergebnis der Vorhersage in predict.csv speichern
prediction = pd.DataFrame()
id = []
for i in range(len(testX)):
    id.append(i)
    i = i + 1
prediction["Id (String)"] = id 
prediction["Category (String)"] = model.predict(testX).astype(int)
print(prediction[:10])
prediction.to_csv("predict.csv", index=False)