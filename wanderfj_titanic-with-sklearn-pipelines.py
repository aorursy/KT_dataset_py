# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plots

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Leitura dos dados

train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')



# Extrair coluna 'Survived' e excluir ela do dataset de treinamento

target = train_df['Survived'].copy()

train_df.drop(['Survived'], axis=1, inplace=True)
# Dimensões do conjunto de treinamento

print(f"n (features): \t{train_df.shape[1]}")

print(f"m (exemplos): \t{train_df.shape[0]}")
# Visualização de alguns dos dados

train_df.head()
# Analise percentual de valores faltantes

(train_df.isnull().sum() / train_df.shape[0]).sort_values(ascending=False)
# Distribuição estatística dos dados

train_df.describe(include='all')
# Histograma das variáveis numéricas

train_df.hist(figsize=(10,8));
# Pipeline para tratamento dos dados numéricos e categóricos



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



features_numericas = ['Age', 'Fare', 'SibSp', 'Parch']

features_categoricas = ['Embarked', 'Sex', 'Pclass']

features_para_remover = ['Name', 'Cabin', 'Ticket', 'PassengerId']



numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', MinMaxScaler())])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder())])



preprocessor = ColumnTransformer(

    transformers=[

        ('Features numericas', numeric_transformer, features_numericas),

        ('Features categoricas', categorical_transformer, features_categoricas),

        ('Feature para remover', 'drop', features_para_remover)

])
# Classificador

from sklearn.linear_model import LogisticRegression



# Modelo de Regressão Logística

model = Pipeline([('preprocessor', preprocessor),

                 ('clf', LogisticRegression(solver='liblinear')),

                 ])



model.fit(train_df, target)



# Verificar a acurácia do modelo

acuracia = round(model.score(train_df, target) * 100, 2)

print(f"Acurácia do modelo: {acuracia}%")
# Aplicação do modelo aos dados de teste

y_pred = model.predict(test_df)



submission = pd.DataFrame({

    "PassengerId": test_df['PassengerId'],

    "Survived": y_pred

})



# Gerar arquivo csv para submissão

submission.to_csv('./submission.csv', index=False)