import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

print(os.listdir("../input/"))



data = pd.read_csv('../input/Entradas Binrias NAP 2-IA_ Medidor de Aceitao Social - Respostas ao formulrio 1.csv')
data.head()
data = data.drop(['Unnamed: 0'],axis=1,)

data.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["Você frequenta este local com muita frequência?"] = labelencoder.fit_transform(data["Você frequenta este local com muita frequência?"].fillna('0'))

data.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data["Escolha um rolê que você mais frequenta ou que mais combine com você:"] = labelencoder.fit_transform(data["Escolha um rolê que você mais frequenta ou que mais combine com você:"].fillna('0'))

data.head()
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Qual a sua idade?', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Em relação ao seu gênero, você é:', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Em relação à orientação sexual, você se considera:', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Em relação à sua residência, ela se localiza:', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Você utiliza o transporte público frequentemente para se deslocar até os rolês?', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'E utiliza também os aplicativos de transporte (Ex: Uber, 99, etc)?', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Você já cursou ou ainda está cursando algum curso de graduação?', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Você trabalha ou exerce atividade remunerada?', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Você frequenta este local com muita frequência?', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Você se autodeclara branco ou branca?', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Você acha que seu estilo de roupa, cabelo ou acessórios está configurado com o padrão socialmente estabelecido?', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Você costuma consumir comidas durante o rolê?', bins=20)
g = sns.FacetGrid( data, col='Escolha um rolê que você mais frequenta ou que mais combine com você:')

g.map(plt.hist, 'Costuma consumir bebidas alcoólicas?', bins=20)

data.head()


classe = data['Escolha um rolê que você mais frequenta ou que mais combine com você:']

atributos = data.drop('Você se autodeclara branco ou branca?', axis=1)

atributos.head()
from sklearn.model_selection import train_test_split

atributos_train, atributos_test, class_train, class_test = train_test_split(atributos, classe, test_size = 0.25 )



atributos_train.describe()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, random_state =0)

model = dtree.fit(atributos_train, class_train)
from sklearn.metrics import accuracy_score

classe_pred = model.predict(atributos_test)

acc = accuracy_score(class_test, classe_pred),

print("My Decision Tree acc is {}".format(acc))