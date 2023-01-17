import pandas as pd

import numpy as np

import scipy

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

import folium

%matplotlib inline







import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 999)
%time train_data = pd.read_csv('../input/titanic/train.csv')

%time test_data = pd.read_csv('../input/titanic/test.csv')
train_data.head()
test_data.head()
train_data.info()
train_data.describe()
train_data.Sex.value_counts()
sns.countplot(y=train_data.Sex ,data=train_data)

plt.xlabel("Contagem")

plt.ylabel("Sexo")

plt.show()
sns.countplot(train_data['Survived'], palette=sns.light_palette("navy", reverse=True))
survived = 'Sobreviveu'

not_survived = 'Não Sobreviveu'



fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))



df_homem = train_data[train_data['Sex'] == 'male']

df_mulher = train_data[train_data['Sex'] == 'female']





ax = sns.distplot(df_homem[df_homem['Survived']==0].Age.dropna(), bins=18, label = 'homens',ax = axes[0], kde =False)

ax = sns.distplot(df_mulher[df_mulher['Survived']==0].Age.dropna(), bins=30, label = 'Mulhres',ax = axes[0], kde =False)



ax.legend()

ax.set_title('Sobrevivente entre os sexo')



ax = sns.distplot(df_homem[df_homem['Survived']==1].Age.dropna(), bins=18, label = 'homens',ax = axes[1], kde =False)

ax = sns.distplot(df_mulher[df_mulher['Survived']==1].Age.dropna(), bins=30, label = 'Mulhres',ax = axes[1], kde =False)





ax.legend()

ax.set_title('Não sobreviventes por sexo')
train_data.Pclass.value_counts()
sns.countplot(train_data['Pclass'], palette=sns.light_palette("navy", reverse=True))
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))



class_1 = train_data[train_data['Pclass'] == 1]

class_2 = train_data[train_data['Pclass'] == 2]

class_3 = train_data[train_data['Pclass'] == 3]







ax = sns.distplot(class_1[class_1['Survived']==0].Age.dropna(), bins=30, label = 'class_1',ax = axes[0], kde =False)

ax = sns.distplot(class_2[class_2['Survived']==0].Age.dropna(), bins=30, label = 'class_2',ax = axes[0], kde =False)

ax = sns.distplot(class_3[class_3['Survived']==0].Age.dropna(), bins=30, label = 'class_3',ax = axes[0], kde =False)



ax.legend()

ax.set_title('Sobrevivente entre por Classe')







ax = sns.distplot(class_1[class_1['Survived']==1].Age.dropna(), bins=30, label = 'class_1',ax = axes[1], kde =False)

ax = sns.distplot(class_2[class_2['Survived']==1].Age.dropna(), bins=30, label = 'class_2',ax = axes[1], kde =False)

ax = sns.distplot(class_3[class_3['Survived']==1].Age.dropna(), bins=30, label = 'class_3',ax = axes[1], kde =False)





ax.legend()

ax.set_title('Não sobreviventes por Classe')
class_sex = sns.FacetGrid(train_data, col='Sex', row='Pclass',  size=2.2, aspect=1.6)

class_sex.map(plt.hist, 'Age', alpha=.5, bins=30, color="b")

class_sex.add_legend();
train_data.SibSp.value_counts()
sns.countplot(train_data['SibSp'], palette=sns.light_palette("navy", reverse=True))
sobreviventes = train_data[train_data['Survived'] == 0]
sobreviventes.SibSp.value_counts()
sns.countplot(sobreviventes['SibSp'], palette=sns.light_palette("navy", reverse=True))
nao_sobreviventes = train_data[train_data['Survived'] == 1]
nao_sobreviventes.SibSp.value_counts()
sns.countplot(nao_sobreviventes['SibSp'], palette=sns.light_palette("navy", reverse=True))
train_data.Parch.value_counts()
sns.countplot(train_data['Parch'], palette=sns.color_palette("GnBu_d"))
sobreviventes_parch = train_data[train_data['Survived'] == 0]
sobreviventes_parch.Parch.value_counts()
sns.countplot(sobreviventes_parch['Parch'], palette=sns.color_palette("GnBu_d"))
sobreviventes_nao_parch = train_data[train_data['Survived'] == 1]
sobreviventes_nao_parch.Parch.value_counts()
sns.countplot(sobreviventes_nao_parch['Parch'], palette=sns.color_palette("GnBu_d"))
train_data.info()
train_data.Age.value_counts()
train_data['Age'].isnull().sum()
train_data.Age.plot.hist(bins=30)
data = [train_data, test_data]



for dataset in data:

    mean = train_data["Age"].mean()

    std = test_data["Age"].std()

    is_null = dataset["Age"].isnull().sum()

   

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_data["Age"].astype(int)
train_data['Age'].isnull().sum()
train_data.Age.plot.hist(bins=30)
train_data.info()
train_data.Embarked.value_counts()
train_data['Embarked'].isnull().sum()
train_data['Embarked'] = train_data['Embarked'].replace(np.nan, 'S')
train_data.Embarked.value_counts()
train_data['Embarked'].isnull().sum()
train_data.Embarked = train_data.Embarked.replace({

    'S': 0,

    'C': 1,

    'Q': 2

})
train_data.Embarked.value_counts()
train_data.Sex = train_data.Sex.replace({

    'male': 0,

    'female': 1

})
train_data.Sex.value_counts()
train_data['Fare'] = train_data.Fare.astype('int')
train_data.info()
train_data = train_data.drop(['PassengerId'], axis=1)
train_data = train_data.drop(['Name'], axis=1)

test_data = test_data.drop(['Name'], axis=1)
train_data = train_data.drop(['Ticket'], axis=1)

test_data = test_data.drop(['Ticket'], axis=1)
train_data = train_data.drop(['Cabin'], axis=1)

test_data = test_data.drop(['Cabin'], axis=1)
test_data.info()
test_data.Sex = test_data.Sex.replace({

    'male': 0,

    'female': 1

})
test_data.Embarked = test_data.Embarked.replace({

    'S': 0,

    'C': 1,

    'Q': 2

})
test_data.Fare.value_counts()
test_data['Fare'] = test_data['Fare'].fillna((test_data['Fare'].mean()))
test_data['Fare'] = test_data.Fare.astype('int')
test_data.info()
x_train = train_data.drop("Survived", axis=1)

y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
# importando as bibliotecas dos modelos classificadores

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



# definindo uma lista com todos os modelos

classifiers = [

    KNeighborsClassifier(),

    GaussianNB(),

    LogisticRegression(dual=False,max_iter=5000),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier()]



# rotina para instanciar, predizer e medir os rasultados de todos os modelos

for clf in classifiers:

    # instanciando o modelo

    clf.fit(x_train, y_train)

    # armazenando o nome do modelo na variável name

    name = clf.__class__.__name__

    # imprimindo o nome do modelo

    print("="*30)

    print(name)

    # imprimindo os resultados do modelo

    print('****Results****')

    y_pred = clf.predict(x_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    print("Precision:", metrics.precision_score(y_test, y_pred))

    print("Recall:", metrics.recall_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score



rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, x_train, y_train, cv=10, scoring = "accuracy")
print("Score:", scores)

print("Média:", scores.mean())

print("Desvio Padrão:", scores.std())