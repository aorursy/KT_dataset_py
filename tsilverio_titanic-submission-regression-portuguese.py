! pip install pyjanitor
! pip install imblearn
#IMPORTANDO BIBLIOTECAS

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



from sklearn import metrics

from sklearn.model_selection import train_test_split

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import janitor

from imblearn.over_sampling import SMOTE



pd.set_option('display.max_rows', None)

params = {'legend.fontsize': 'x-large',

          'figure.figsize': (16, 8),

          'axes.labelsize': 'x-large',

          'axes.titlesize':'x-large',

          'xtick.labelsize':'x-large',

          'ytick.labelsize':'x-large'}



%matplotlib inline

plt.rcParams.update(params)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train=pd.read_csv('/kaggle/input/titanic/train.csv').clean_names()

df_test=pd.read_csv('/kaggle/input/titanic/test.csv').clean_names()
df_train.head()
df_train.info()
df_train.isnull().sum()
df_test.isnull().sum()
#Plotando um grafico de barras dos sobreviventes por Sexo

sns.barplot(x="sex", y="survived", data=df_train)



#Mostrando as % de mulheres vs homens que sobreviveram

print("Porcentagem de Mulheres Sobreviventes:", df_train["survived"][df_train["sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Porcentagem de Homens Sobreviventes:", df_train["survived"][df_train["sex"] == 'male'].value_counts(normalize = True)[1]*100)
#Plotando sobreviventes por Pclass

sns.barplot(x="pclass", y="survived", data=df_train, hue='sex')



#Mostrando a % de sobreviventes pelas Classes.

print("Porcentagem da Pclass = 1 que sobreviveu:", df_train["survived"][df_train["pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Porcentagem da Pclass = 2 que sobreviveu:", df_train["survived"][df_train["pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Porcentagem da Pclass = 3 que sobreviveu:", df_train["survived"][df_train["pclass"] == 3].value_counts(normalize = True)[1]*100)
#Plotando sobreviventes por Sibsp

sns.barplot(x="sibsp", y="survived", data=df_train)



#Mostrando a % de Sobreviventes por Sibsp

print("Porcentagem de SibSp = 0 Sobrevivente:", df_train["survived"][df_train["sibsp"] == 0].value_counts(normalize = True)[1]*100)



print("Porcentagem de SibSp = 1 Sobrevivente:", df_train["survived"][df_train["sibsp"] == 1].value_counts(normalize = True)[1]*100)



print("Porcentagem de SibSp = 2 Sobrevivente:", df_train["survived"][df_train["sibsp"] == 2].value_counts(normalize = True)[1]*100)



print("Porcentagem de SibSp = 3 Sobrevivente:", df_train["survived"][df_train["sibsp"] == 3].value_counts(normalize = True)[1]*100)



print("Porcentagem de SibSp = 4 Sobrevivente:", df_train["survived"][df_train["sibsp"] == 4].value_counts(normalize = True)[1]*100)



#print("Porcentagem de SibSp = 5 Sobrevivente:", df_train["survived"][df_train["sibsp"] == 5].value_counts(normalize = True)[1]*100)
#Plotando sobreviventes pela quantidade de filhos e Pais

sns.barplot(x="parch", y="survived", data=df_train)

plt.show()
sns.distplot(df_train['age'], bins=80);
#Classificando e comparando as Idades dos sobreviventes

#Foi adicionado um valor negativo para os registros sem identificação da idade

df_train["age"] = df_train["age"].fillna(-0.5)

df_test["age"] = df_test["age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Desconhecidos', 'Bebes', 'Crianças', 'Adolescentes', 'Estudantes', 'Jovens Adultos', 'Adultos', 'Idosos']

df_train['agegroup'] = pd.cut(df_train["age"], bins, labels = labels)

df_test['agegroup'] = pd.cut(df_test["age"], bins, labels = labels)



#Plotando os sobreviventes dos grupos que criamos 

sns.barplot(x="agegroup", y="survived", data=df_train, hue='sex')

plt.show()
#Existem muitos registros sem Cabine, vamos criar uma Dummy para classificar aqueles que possuem cabine e qual a relação com os sobreviventes

df_train["cabinbool"] = (df_train["cabin"].notnull().astype('int'))

df_test["cabinbool"] = (df_test["cabin"].notnull().astype('int'))



#Calculando a % de sobreviventes com cabine

print("Porcentagem CabinBool = 1 Sobreviventes:", df_train["survived"][df_train["cabinbool"] == 1].value_counts(normalize = True)[1]*100)

print("Porcentagem CabinBool = 0 Sobreviventes:", df_train["survived"][df_train["cabinbool"] == 0].value_counts(normalize = True)[1]*100)



#Plotando as relação das cabines com sobreviventes

sns.barplot(x="cabinbool", y="survived", data=df_train)

plt.show()
df_test.describe(include="all")

#total de 418 passageiros.

#Temos 1 valor no "Fare" faltando.

#Estão faltando muitos registos de idade
#Dropando a coluna da Cabine, já que não será mais necessária

df_train = df_train.drop(['cabin'], axis = 1)

df_test = df_test.drop(['cabin'], axis = 1)
#Dropando a coluna de Ticket, já que não encontramos uma utilidade

df_train = df_train.drop(['ticket'], axis = 1)

df_test = df_test.drop(['ticket'], axis = 1)
#Preenchendo os valores faltantes no Embarque com os dados do Porto que possuem mais passageiros

print("PAssageiros em Southampton (S):")

southampton = df_train[df_train["embarked"] == "S"].shape[0]

print(southampton)



print("Passageiros em Cherbourg (C):")

cherbourg = df_train[df_train["embarked"] == "C"].shape[0]

print(cherbourg)



print("Passageiros em Queenstown (Q):")

queenstown = df_train[df_train["embarked"] == "Q"].shape[0]

print(queenstown)
#Preenchendo os faltantes com o Embarked S (644 passageiros)

df_train = df_train.fillna({"embarked": "S"})
combine = [df_train, df_test]



#Extraindo os titulos do Nome

for dataset in combine:

    dataset['title'] = dataset.name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(df_train['title'], df_train['sex'])
#Concatenando os titulos por titulos mais comuns

for dataset in combine:

    dataset['title'] = dataset['title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['title'] = dataset['title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['title'] = dataset['title'].replace('Mlle', 'Miss')

    dataset['title'] = dataset['title'].replace('Ms', 'Miss')

    dataset['title'] = dataset['title'].replace('Mme', 'Mrs')



df_train[['title', 'survived']].groupby(['title'], as_index=False).mean()
#Categorizando cada grupo de titulos

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:

    dataset['title'] = dataset['title'].map(title_mapping)

    dataset['title'] = dataset['title'].fillna(0)



df_train.head(n=20)
#Preenchendo as idades desconhecidas com o valor da moda do grupo de titulos

mr_age = df_train[df_train["title"] == 1]["agegroup"].mode() #Jovens Adultos

miss_age = df_train[df_train["title"] == 2]["agegroup"].mode() #Estudantes

mrs_age = df_train[df_train["title"] == 3]["agegroup"].mode() #Jovens Adultos

master_age = df_train[df_train["title"] == 4]["agegroup"].mode() #Adultos

royal_age = df_train[df_train["title"] == 5]["agegroup"].mode() #Adultos

rare_age = df_train[df_train["title"] == 6]["agegroup"].mode() #Adultos



age_title_mapping = {1: "Jovens Adultos", 2: "Estudantes", 3: "Jovens Adultos", 4: "Adultos", 5: "Adultos", 6: "Adultos"}





for x in range(len(df_train["agegroup"])):

    if df_train["agegroup"][x] == "Desconhecidos":

        df_train["agegroup"][x] = age_title_mapping[df_train["title"][x]]

        

for x in range(len(df_test["agegroup"])):

    if df_test["agegroup"][x] == "Desconhecidos":

        df_test["agegroup"][x] = age_title_mapping[df_test["title"][x]]
#Alterando os grupos para categorias

age_mapping = {'Bebes': 1, 'Crianças': 2, 'Adolescentes': 3, 'Estudantes': 4, 'Jovens Adultos': 5, 'Adultos': 6, 'Idosos': 7}

df_train['agegroup'] = df_train['agegroup'].map(age_mapping)

df_test['agegroup'] = df_test['agegroup'].map(age_mapping)



df_train.head()
df_test.head()
#Removendo a coluna idade, já que criamos grupos

df_train = df_train.drop(['age'], axis = 1)

df_test = df_test.drop(['age'], axis = 1)
#Removendo coluna nome, ja que extraimos o titulo que foi utilizado nas idades

df_train = df_train.drop(['name'], axis = 1)

df_test = df_test.drop(['name'], axis = 1)
#Substituindo Sexo por 0 e 1

sex_mapping = {"male": 0, "female": 1}

df_train['sex'] = df_train['sex'].map(sex_mapping)

df_test['sex'] = df_test['sex'].map(sex_mapping)



df_train.head(n=20)
#Substituindo os Portos por valores 1, 2 e 3

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

df_train['embarked'] = df_train['embarked'].map(embarked_mapping)

df_test['embarked'] = df_test['embarked'].map(embarked_mapping)



df_train.head()
df_train.isnull().sum()
df_test.isnull().sum()
#Preenchendo com o valor médio da classe que pertence o registro faltante(Classe 3) - Depois iremos categorizar

for x in range(len(df_test["fare"])):

    if pd.isnull(df_test["fare"][x]):

        pclass = df_test["pclass"][x] #Pclass = 3

        df_test["fare"][x] = round(df_train[df_train["pclass"] == pclass]["fare"].mean(), 4)
df_test.isnull().sum()
#Categorizando os valores do Fare em 4

df_train['fareband'] = pd.qcut(df_train['fare'], 4, labels = [1, 2, 3, 4])

df_test['fareband'] = pd.qcut(df_test['fare'], 4, labels = [1, 2, 3, 4])
#Removendo coluna Fare, ja que não precisamos mais dela

df_train = df_train.drop(['fare'], axis = 1)

df_test = df_test.drop(['fare'], axis = 1)
df_train.head()
df_test.head()
#Criando novas bases de treino e teste

train = df_train

test = df_test
#Removendo colunas

predictors = train.drop(['survived', 'passengerid'], axis=1)

target = train["survived"]

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
#Plotando valores

count_classes = pd.value_counts(y_train, sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Histograma de Doentes")

plt.xlabel("Classe")

plt.ylabel("Frequency")
#Balanceando

sm = SMOTE(random_state=1234)



X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
#Plotando valores dos sobreviventes

count_classes = pd.value_counts(y_train_res, sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Histograma de Doentes")

plt.xlabel("Classe")

plt.ylabel("Frequency")
def CMatrix(CM,labels =['Morreu','Viveu']):

    df = pd.DataFrame( data = CM, index = labels, columns = labels)

    df.index.name ='Real'

    df.columns.name = 'Previsto'

    df.loc['Total']= df.sum()

    df['Total']= df.sum(axis=1)

    return df
#Criando Matriz de Confusão

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix, precision_recall_curve



logistic_regression= LogisticRegression(solver = 'sag', max_iter = 10000)

log = logistic_regression.fit(X_train_res,y_train_res)



y_pred_test_log = logistic_regression.predict(X_test) 

acuracia_log = accuracy_score(y_pred=y_pred_test_log,y_true=y_test)

precisao_log = precision_score(y_pred=y_pred_test_log,y_true=y_test)

recall_log = recall_score(y_pred=y_pred_test_log,y_true=y_test)



CM= confusion_matrix(y_pred=y_pred_test_log,y_true=y_test)

CMatrix(CM)
(98+62)/197
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
#Preparando ids como PassengerId para a predição de sobreviventes 

ids = df_test['passengerid']

predictions = logreg.predict(test.drop('passengerid', axis=1))



#Convertendo e exportando o csv file com o nome submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)
files.download('submission.csv')
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)

CM= confusion_matrix(y_pred=y_pred,y_true=y_test)

CMatrix(CM)
(110+57)/197
#set ids as PassengerId and predict survival 

ids = df_test['passengerid']

predictions = gbk.predict(test.drop('passengerid', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission_gbk.csv', index=False)
files.download('submission_gbk.csv')