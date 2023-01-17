import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

print(os.listdir("../input"))



data = pd.read_csv('../input/Anlise de vagas de T.I.csv')
data.head()
data = data.drop(['Carimbo de data/hora'],axis=1,)

data.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["A oferta de trabalho é referente a que área de T.I?"] = labelencoder.fit_transform(data["A oferta de trabalho é referente a que área de T.I?"].fillna('0'))

data.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["Nesta oferta de trabalho, está sendo exigido experiência?"] = labelencoder.fit_transform(data["Nesta oferta de trabalho, está sendo exigido experiência?"].fillna('0'))

data.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["Qual(quais) conhecimento(s) de tecnologia está sendo exigido na oferta de trabalho?"] = labelencoder.fit_transform(data["Qual(quais) conhecimento(s) de tecnologia está sendo exigido na oferta de trabalho?"].fillna('0'))

data.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["Nesta oferta de trabalho, está sendo exigido certificação? Se sim, quais?"] = labelencoder.fit_transform(data["Nesta oferta de trabalho, está sendo exigido certificação? Se sim, quais?"].fillna('0'))

data.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["A oferta de trabalho é para qual região do país?"] = labelencoder.fit_transform(data["A oferta de trabalho é para qual região do país?"].fillna('0'))

data.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["Para ocupar está vaga, é exigido ensino superior?"] = labelencoder.fit_transform(data["Para ocupar está vaga, é exigido ensino superior?"].fillna('0'))

data.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["Está vaga é referente a que data?"] = labelencoder.fit_transform(data["Está vaga é referente a que data?"].fillna('0'))

data.head()
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='Qual a média salarial proposta?')

g.map(plt.hist, 'A oferta de trabalho é referente a que área de T.I?', bins=20)
classe = data['Qual a média salarial proposta?']

atributos = data.drop('Qual a média salarial proposta?', axis=1)

atributos.head()
from sklearn.model_selection import train_test_split

atributos_train, atributos_test, class_train, class_test = train_test_split(atributos, classe, test_size = 0.25 )



atributos_train.describe()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, random_state =0)

model = dtree.fit(atributos_train, class_train)
from sklearn.metrics import accuracy_score

classe_pred = model.predict(atributos_test)

acc = accuracy_score(class_test, classe_pred)

print("My Decision Tree acc is {}".format(acc))