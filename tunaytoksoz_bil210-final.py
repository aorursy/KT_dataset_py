import numpy as np

import pandas as pd 

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import recall_score, f1_score, precision_score

from sklearn.tree import DecisionTreeClassifier

from warnings import filterwarnings

import matplotlib.pyplot as plt

from sklearn import ensemble

from sklearn.metrics import confusion_matrix as cm

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.naive_bayes import GaussianNB

filterwarnings('ignore')
df = pd.read_csv("../input/tabletcsv/tablet.csv")
df.shape
df.head()
df.info()
df.describe().T
df.isna().sum()
corr = df.corr()

corr
sns.set(rc={'figure.figsize':(10,8)})

sns.heatmap(corr,

           xticklabels=corr.columns,

           yticklabels=corr.columns,annot=True,vmin=-1, vmax=1, fmt=".2g");
sns.scatterplot(x = "ArkaKameraMP", y = "OnKameraMP", data = df);
sns.scatterplot(x = "CozunurlukGenislik", y = "CozunurlukYükseklik", data = df);
sns.distplot(df["RAM"], bins=16, color="green");
sns.jointplot(x = df["RAM"], y = df["BataryaGucu"], kind = "kde", color = "purple");
sns.barplot(x = "MikroislemciHizi", y = "FiyatAraligi", data = df);
sns.barplot(x = "Agirlik", y = "FiyatAraligi", data = df);
sns.barplot(x = "CozunurlukYükseklik", y = "FiyatAraligi", data = df);
sns.barplot(x = "CozunurlukGenislik", y = "FiyatAraligi", data = df);
sns.barplot(x = "RAM", y = "FiyatAraligi", data = df); 
sns.catplot(x = "RAM", y = "FiyatAraligi", data = df);
sns.barplot(x = "DahiliBellek", y = "FiyatAraligi", data = df);
sns.barplot(x = "BataryaOmru", y = "FiyatAraligi", data = df);
sns.barplot(x = "BataryaGucu", y = "FiyatAraligi", data = df);
df.head()
df['DortG']=df['4G']

df['UcG']=df['3G']

df['Bluetooth'] = df.Bluetooth.map({'Yok':0, 'Var':1})

df['CiftHat'] = df.CiftHat.map({'Yok':0, 'Var':1})

df['4G'] = df.DortG.map({'Yok':0, 'Var':1})

df['3G'] = df.UcG.map({'Yok':0, 'Var':1})

df['Dokunmatik'] = df.Dokunmatik.map({'Yok':0, 'Var':1})

df['WiFi'] = df.WiFi.map({'Yok':0, 'Var':1})



df.drop(['Renk','DortG','UcG'], axis=1,inplace=True)

df.head() #Burada değerlerin değiştiğini görebiliyoruz.
df.info() #Burada da değerlerin tiplerinin değiştiğini görebiliyoruz.
df['RAM'].fillna(df["RAM"].mean(),inplace=True)

df['OnKameraMP'].fillna(df["OnKameraMP"].mean(),inplace=True)

df.info()
df.isna().sum()
y = df['FiyatAraligi']

X = df.drop(['FiyatAraligi'], axis=1)
y
X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train
X_test
y_train
y_test
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)
nb_model
nb_predict = nb_model.predict(X_test)

nb_predict
cm = confusion_matrix(y_test,nb_predict)

cm
accuracy = accuracy_score(y_test, nb_predict)

accuracy
corr = df.corr()

corr
sns.set(rc={'figure.figsize':(15,10)})

sns.heatmap(corr,

           xticklabels=corr.columns,

           yticklabels=corr.columns,annot=True,vmin=-1, vmax=1, fmt=".2g");
cart = DecisionTreeClassifier(random_state = 42, criterion='gini')

cart_model = cart.fit(X_train, y_train)
cart_model
df.columns
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred) #Başarı puanımızda artış var.
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
print(classification_report(y_test, y_pred))
ranking = cart.feature_importances_

features = np.argsort(ranking)[::-1][:18]

columns = X.columns



plt.figure(figsize = (16, 9))

plt.title("Karar Ağacına Göre Özniteliklerin Önem Derecesi", size = 18)

plt.bar(range(len(features)), ranking[features], color="lime", align="center")

plt.xticks(range(len(features)), columns[features], rotation=80)

plt.show()
knn_params = {"n_neighbors": np.arange(2,15)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 3)

knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))

print("En iyi parametreler: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(10)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred) #Başarı puanımızda önemli bir artış var.
score_list = []



for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(X_train,y_train)

    score_list.append(knn2.score(X_test, y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k değerleri")

plt.ylabel("doğruluk skoru")

plt.show()
cross_val_score(knn_tuned, X_test, y_test, cv = 10)
cross_val_score(cart_model, X, y, cv = 10)
print(classification_report(y_test, y_pred))