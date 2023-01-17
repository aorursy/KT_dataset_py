import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score, cross_val_predict

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 

from sklearn import model_selection

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import BaggingRegressor

from argparse import Namespace

import seaborn as sns 

from sklearn import preprocessing

from sklearn.metrics import roc_auc_score, roc_curve, recall_score, f1_score, precision_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("../input/dataaa/tablet.csv")
df.head()
df.info()
df.isnull().sum()
df.dtypes
df.describe().T
sns.distplot(df["BataryaGucu"], bins=16, color="purple");
sns.distplot(df["MikroislemciHizi"], bins=16, color="purple");
sns.distplot(df["OnKameraMP"], bins=16, color="purple");
sns.distplot(df["DahiliBellek"], bins=16, color="purple");
sns.distplot(df["Kalinlik"], bins=16, color="purple");
sns.distplot(df["Agirlik"], bins=16, color="purple");
sns.distplot(df["CekirdekSayisi"], bins=16, color="purple");
sns.distplot(df["ArkaKameraMP"], bins=16, color="purple");
sns.distplot(df["CozunurlukYükseklik"], bins=16, color="purple");
sns.distplot(df["CozunurlukGenislik"], bins=16, color="purple");
sns.distplot(df["RAM"], bins=16, color="purple");
sns.distplot(df["BataryaOmru"], bins=16, color="purple");
df.cov()
df.corr() 
sns.countplot(x = "Bluetooth", data = df);
sns.countplot(x = "CiftHat", data = df);
sns.countplot(x = "4G", data = df);
sns.countplot(x = "3G", data = df);
sns.countplot(x = "Dokunmatik", data = df);
sns.countplot(x = "WiFi", data = df);
sns.countplot(x = "FiyatAraligi", data = df);
sns.countplot(x = "Renk", data = df);
df[df["RAM"].isnull()]
df[df["OnKameraMP"].isnull()]
sns.countplot(df[df["RAM"].isnull()]["FiyatAraligi"]);
sns.countplot(df[df["RAM"].isnull()]["4G"]);
sns.countplot(df[df["RAM"].isnull()]["3G"]);
sns.countplot(df[df["RAM"].isnull()]["CiftHat"]);
sns.countplot(df[df["RAM"].isnull()]["WiFi"]);
sns.countplot(df[df["RAM"].isnull()]["Dokunmatik"]);
sns.countplot(df[df["RAM"].isnull()]["Renk"]);
df.groupby("FiyatAraligi")[["RAM"]].mean()
Pahalı_Ramler = df[(df["FiyatAraligi"] == "Pahalı") & (df["RAM"].isnull())].index

Pahalı_Ramler
df.loc[Pahalı_Ramler ,"RAM"] = 3449
df.isna().sum()["RAM"]
sns.countplot(df[df["OnKameraMP"].isnull()]["FiyatAraligi"]);
sns.countplot(df[df["OnKameraMP"].isnull()]["4G"]);
sns.countplot(df[df["OnKameraMP"].isnull()]["3G"]);
sns.countplot(df[df["OnKameraMP"].isnull()]["CiftHat"]);
sns.countplot(df[df["OnKameraMP"].isnull()]["WiFi"]);
sns.countplot(df[df["OnKameraMP"].isnull()]["Dokunmatik"]);
sns.countplot(df[df["OnKameraMP"].isnull()]["Renk"]);
df.groupby("FiyatAraligi")[["OnKameraMP"]].mean()
ucuz_mpler = df[(df["FiyatAraligi"] == "Çok Ucuz") & (df["OnKameraMP"].isnull())].index

ucuz_mpler
df.loc[ucuz_mpler ,"OnKameraMP"] = 4
df.isna().sum()["OnKameraMP"]
label_encoder = preprocessing.LabelEncoder()
df['CiftHat'] = label_encoder.fit_transform(df['CiftHat'])

df.head()
df['Bluetooth'] = label_encoder.fit_transform(df['Bluetooth'])

df.head()
df['4G'] = label_encoder.fit_transform(df['4G'])

df.head()
df['3G'] = label_encoder.fit_transform(df['3G'])

df.head()
df['Dokunmatik'] = label_encoder.fit_transform(df['Dokunmatik'])

df.head()
df['WiFi'] = label_encoder.fit_transform(df['WiFi'])

df.head()
df["Renk"].unique()
df['Renk'] = pd.Categorical(df['Renk'])

dfDummies = pd.get_dummies(df['Renk'], prefix = 'Renk')

dfDummies
df = pd.concat([df, dfDummies], axis=1)

df.head()
df = df.drop(['Renk'], axis=1)
y = df['FiyatAraligi']

X = df.drop(['FiyatAraligi'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y, 

                                                    test_size = 0.25, 

                                                    random_state = 42)
X_train
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb_model = nb.fit(X_train, y_train)
nb_model
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier(random_state = 42, criterion='gini')

cart_model = cart.fit(X_train, y_train)
cart_model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn_model = knn.fit(X_train, y_train)
knn_model
y_pred = nb_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
print(classification_report(y_test, y_pred))
y_pred = knn_model.predict(X)
accuracy_score(y, y_pred)
karmasiklik_matrisi = confusion_matrix(y, y_pred)

print(karmasiklik_matrisi)
print(classification_report(y, y_pred))
cart = DecisionTreeClassifier(random_state = 42, criterion='entropy')

cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
karmasiklik_matrisi = confusion_matrix(y_test, y_pred)

print(karmasiklik_matrisi)
print(classification_report(y_test, y_pred))
knn_params = {"n_neighbors": np.arange(2,15)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, knn_params, cv = 3)

knn_cv.fit(X_train, y_train)
print("En iyi skor: " + str(knn_cv.best_score_))

print("En iyi parametreler: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(9)

knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
score_list = []



for each in range(2,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(X_train,y_train)

    score_list.append(knn2.score(X_test, y_test))

    

plt.plot(range(2,15),score_list)

plt.xlabel("k değerleri")

plt.ylabel("doğruluk skoru")

plt.show()