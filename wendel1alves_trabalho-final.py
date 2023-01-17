# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
# identificando as colunas que são object

df.info()
#identificando os valores da coluna REASON - object

df['REASON'].value_counts()
#identificando os valores da coluna JOB - object



df['JOB'].value_counts()
# Alterando valores da coluna REASON

#0 DebtCon

#1 HomeImp



reasonvalor = {'DebtCon': 0, 'HomeImp': 1}



df['REASON'] = df['REASON'].replace(reasonvalor)

# Alterando valores da coluna JOB

#0 Other

#1 ProfExe

#2 Office

#3 Mgr

#4 Self

#5 Sales



trabalho = {'Other': 0, 'ProfExe': 1, 'Office': 2,'Mgr': 3, 'Self': 4, 'Sales': 5}



df['JOB'] = df['JOB'].replace(trabalho)
#verificando se ainda existem colunas do tipo object



df.select_dtypes('object').head()

#verificando valores nulos

df.isnull().sum()
df.describe()
#analisando a variável target

df['BAD'].value_counts().reset_index()
df['MORTDUE'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt
#Traçando gráficos para analisar as variáveis

plt.figure(figsize=(4, 4))

sns.countplot(y="REASON", data=df)



plt.figure(figsize=(8, 8))

sns.countplot(y="JOB", data=df)



plt.figure(figsize=(8, 8))

sns.countplot(y="DEROG", data=df)



plt.figure(figsize=(8, 8))

sns.countplot(y="DELINQ", data=df)



plt.figure(figsize=(8, 8))

sns.countplot(y="NINQ", data=df)
#Traçando gráficos para analisar as variáveis



df[['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'CLAGE', 'CLNO', 'DEBTINC' ]].hist(figsize=[20,20])
# conhecendo os valores de algumas variáveis

#criando novo dataframe para imputar valores nulos



df2=df
df2
#Excluindo valores onde só possuem os dados das variáveis BAD, LOAN e mais alguma outra



#Excluindo variáveis com dados de BAD e LOAN

df2 = df2.dropna(subset=['MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e MORTDUE

df2 = df2.dropna(subset=['VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e VALUE

df2 = df2.dropna(subset=['MORTDUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e REASON

df2 = df2.dropna(subset=['MORTDUE', 'VALUE', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e JOB

df2 = df2.dropna(subset=['MORTDUE', 'VALUE', 'REASON', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e 'YOJ'

df2 = df2.dropna(subset=['MORTDUE', 'VALUE', 'REASON', 'JOB', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e DEROG

df2 = df2.dropna(subset=['MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e DELINQ

df2 = df2.dropna(subset=['MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e CLAGE

df2 = df2.dropna(subset=['MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'NINQ', 'CLNO', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e NINQ

df2 = df2.dropna(subset=['MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'CLNO', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e CLNO

df2 = df2.dropna(subset=['MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'DEBTINC'], how='all')



#Excluindo variáveis com dados de BAD, LOAN e DEBTINC

df2 = df2.dropna(subset=['MORTDUE', 'VALUE', 'REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO'], how='all')
df2
#criando nova coluna com a relação entre a coluna VALUE e a coluna MORTDUE



df2['VALUEMORTDUE'] = df2['VALUE'] / df2['MORTDUE']
df2.head(30)
#verificando a moda (valor mais frequente) para definir a imputação nos valores nulos das colunas VALUE e MORTDUE

mode = float(df2['VALUEMORTDUE'].mode())



mode
#Criando um diconário com os valores correspondentes a proporção média do dataframe e imputando na base substituindo apenas os valores vazios

MORTDUEMEDIA = {'MORTDUE': df2['VALUE']/mode}

df2 = df2.fillna(value=MORTDUEMEDIA)





VALUEMEDIA = {'VALUE': df2['MORTDUE']*mode}

df2=df2.fillna(value=VALUEMEDIA)

df2.info()
df2.isnull().sum()
#Rodando VALUEMORDUE novamente 

df2['VALUEMORTDUE'] = df2['VALUE'] / df2['MORTDUE']
df['MORTDUE'].min()
df2.info()
df2.isnull().sum()
import plotly.io as pio



#GRÁFICO PARA VERIFICAR EM QUAIS CAMPOS DE JOB ESTÃO OS MISSING DE REASON

GRAREASON = [dict(

  type = 'bar',

  x = df2['JOB'],

  y = df2['REASON'].value_counts(dropna=False),

  mode = 'markers',

#  transforms = [dict(

#    type = 'filter',

#    target = 'y',

#    operation = '==',

#    value = NaN

#  )]

)]



layout = dict(

    title = 'REASON = 1'

)



fig_dict = dict(data=GRAREASON, layout=layout)



pio.show(fig_dict, validate=False)





#GRÁFICO PARA VERIFICAR EM QUAIS CAMPOS DE JOB ESTÃO A MAIORIA DOS DADOS DE REASON

import plotly.express as px

fig = px.bar(df2, x="REASON", y="JOB", color='JOB')

fig.show()
df2.isnull().sum()
#IMPUTANDO VALORES MISSING PELA MODA OU MÉDIA



k = ['REASON', 'JOB', 'YOJ', 'DEROG', 'DELINQ', 'NINQ', 'CLNO' ]



for i in list (k):

    mode = float(df2[i].mode())

    df2[i] = df2[i].fillna(mode)

    



W = ['MORTDUE', 'VALUE', 'CLAGE', 'DEBTINC']

for i in list (W):

    mean = df2[W].mean()

    df2[W] = df2[W].fillna(mean)

    

df2['VALUEMORTDUE'] = df2['VALUE'] / df2['MORTDUE']


df2.isnull().sum()
#Verificando a correlação entre as variáveis



corr = df2.corr(method='spearman')

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 18))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.30, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Separando o dataframe



# Importando o train_test_split

from sklearn.model_selection import train_test_split



# Separando treino e teste

train, test = train_test_split(df2, test_size=0.20, random_state=42)



# Não vamos mais usar o dataset de validação



train.shape, test.shape
# definindo colunas de entrada



feats = [c for c in df.columns if c not in ['BAD']]
# Random Forest



# Importando o modelo

from sklearn.ensemble import RandomForestClassifier



# Instanciar o modelo

rf = RandomForestClassifier(n_estimators=200, random_state=42)
# Usar o cross validation

from sklearn.model_selection import cross_val_score



scores = cross_val_score(rf, train[feats], train['BAD'], n_jobs=-1, cv=5)



scores, scores.mean()
# XGBoost



# Importar o modelo

from xgboost import XGBClassifier



# Instanciar o modelo

xgb = XGBClassifier(n_estimators=200, n_jobs=-1, random_state=42, learning_rate=0.05)
# Usando o cross validation

scores = cross_val_score(xgb, train[feats], train['BAD'], n_jobs=-1, cv=5)



scores, scores.mean()

# Usando o Randon Forest para treinamento e predição

rf.fit(train[feats], train['BAD'])
# Fazendo predições

preds = rf.predict(test[feats])
# Medir o desempenho do modelo

from sklearn.metrics import accuracy_score



accuracy_score(test['BAD'], preds)
# Feature Importance

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()

cnf_matrix = metrics.confusion_matrix(test['BAD'], preds)





%matplotlib inline

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Matriz Confusão', y=1.1)

plt.ylabel('Real')

plt.xlabel('Predito')