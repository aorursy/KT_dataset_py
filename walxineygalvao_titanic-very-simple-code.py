# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Leitura de dados de train e criação de DataFrame

data_train = pd.read_csv("../input/titanic/train.csv")

df_train = pd.DataFrame(data_train)

df_train.head()
# Leitura de dados de test e criação de DataFrame

data_test = pd.read_csv("../input/titanic/test.csv")

df_test = pd.DataFrame(data_test)

df_test.head()
# não utilizei esses dados

data_gender = pd.read_csv("../input/titanic/gender_submission.csv")

df_gender = pd.DataFrame(data_gender)

df_gender.head()
# Aqui é posivel verificar que o datase de train possui 891 linhas e possui algumas colunas com nulos 

df_train.info()
# Aqui é posivel verificar que o datase de test possui 418 linhas e possui algumas colunas com nulos 

df_test.info()
# Com a função isnull() é possivel ver realmente quantos nulos tem agregando a função a sum()

df_train.isnull().sum()
# Com a função isnull() é possivel ver realmente quantos nulos tem agregando a função a sum()

df_test.isnull().sum()
# A coluna de Sex será transformada em dummy, fazendo binario 1 = female e 0 = male

# Estrou efetuando o drop de uma das colunas visto que não é necessário ter as duas colunas

df_train = pd.get_dummies(df_train, columns=['Sex'])
# A coluna de Sex será transformada em dummy, fazendo binario 1 = female e 0 = male

# Estrou efetuando o drop de uma das colunas visto que não é necessário ter as duas colunas

df_test = pd.get_dummies(df_test, columns=['Sex'])
# A coluna Sex foi transformada em duas Sex_female e Sex_male

df_train.head()
# A coluna Sex foi transformada em duas Sex_female e Sex_male

df_test.head()
# Efetuando o drop da coluna Sex_male no dataset train

df_train = df_train.drop(['Sex_male'], axis = 1)

# Efetuando o drop da coluna Sex_male no dataset test

df_test = df_test.drop(['Sex_male'], axis = 1)
# Apenas visualizando o resultado

#df_test.head()

df_train.head()
# importando a lib statistics para usar a função mode (moda)

from statistics import mode
# Efetuando o preenchimento dos nulos do dataset train com valores a moda, ou seja o que mais repete

df_train['Embarked'] = df_train['Embarked'].fillna(mode(df_train['Embarked']))
# Efetuando o preenchimento dos nulos do dataset test com valores a moda, ou seja o que mais repete

df_test['Embarked'] = df_test['Embarked'].fillna(mode(df_test['Embarked']))
# Verificando os nulos novamente no dataset train

df_train.isnull().sum()

# Ainda ha nulos em Age e em Cabin
# Verificando os nulos novamente no dataset test

df_test.isnull().sum()

# Ainda ha nulos em Age e em Cabin
# Visualizando quantas variações temos nos dados de Embarked para poder transformar em dummies

print(df_test['Embarked'].unique())

print(df_train['Embarked'].unique())
# Transformando em dummy a coluna Embarked

df_train = pd.get_dummies(df_train, columns=['Embarked'])

df_test = pd.get_dummies(df_test, columns=['Embarked'])
# Aqui é possivel ver que a coluna Embarked do dataset train se trasnformou em três colunas C, Q e S

df_train.head()
# Aqui é possivel ver que a coluna Embarked do dataset test se trasnformou em três colunas C, Q e S

df_test.head()
# Verificando quantos tem e cada, eu vou optar por remover a coluna que tem menos 1 no cxaso a Q

print(df_train['Embarked_C'].value_counts())

print(df_train['Embarked_Q'].value_counts())

print(df_train['Embarked_S'].value_counts())
# Deletando uma das colunas dummy, visto que temos agora C,Q e S,

# Exemnplo...

# Se for C a coluna C estara com 1 e as demais com 0 se for Q estara com 1 eo restante 0 e se for S terá 1 eo restante 0

# portanto evitar a armadilha das dummies é necessario apagar uma das colunas

# se fo C terá 1 se for S terá 1 e se não for nenhum deles ambos estará zero, automaticamente sendo Q.

df_train = df_train.drop(['Embarked_Q'], axis = 1)

df_test = df_test.drop(['Embarked_Q'], axis = 1)
df_train.head()
df_test.head()

# Um exemplo é a primeira linha onde Sex_female esta 0 isso implica em ser Sex_male

# assim como Embarked_C e Embarked_S estão zero implica em ser Embarked_Q
# Preenchendo os nulos da coluna Age com os valores da media.

#df_train['Age'] = df_train.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))

df_train['Age'] = df_train['Age'].transform(lambda x: x.fillna(x.median()))
# Visualizando os nulos novamente

df_train.isnull().sum()
# Preenchendo os nulos da coluna Age com os valores da media.

df_test['Age'] = df_test['Age'].transform(lambda x: x.fillna(x.median()))

# Visualizando os nulos novamente

df_test.isnull().sum()
# porém o dataset de test ainda temum nulo na coluna Fare que será preenchida com a média

# Preenchendo os nulos da coluna Age com os valores da media.

df_test['Fare'] = df_test['Fare'].transform(lambda x: x.fillna(x.median()))

# Visualizando os nulos novamente

df_test.isnull().sum()
# Preenchendo os nulos com valores U, na coluna Cabin

df_train['Cabin'] = df_train['Cabin'].fillna('U')

df_test['Cabin'] = df_test['Cabin'].fillna('U')
# visualizando os dados, na coluna Cabin os nulos foram preenchidos com U

#df_train.head()

df_test.head()
# importando a lib de regex para poder utilizar na obtenção das letras da coluna Cabin...

# pois ainda nao podemos utilizar essa coluna no dataset, pois não é numero, então anlisando a coluna

# podemos ver que é composta por letras e numeros mas as letras se repetem assim traçando algum padrão

# então vamos ficar apenas com as letras...

import re
# Aqui é possivel ver um exemplo de Cabin

df_train['Cabin'].unique()
# Esta função lambda faz uso de regex para localizar as letras de cada linha da coluna Cabin, e agrupa-las

df_train['Cabin'] = df_train['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())
# listando as letras da coluna Cabin

df_train['Cabin'].unique().tolist()
# Criando um dicionario para numerar as letras extraídas da coluna Cabin

# poderia ter transformado em dummies mas isto iria aumentar as colunas do meu dataset então apenas numerei

# o mnais correto seria fazer uma função pra percorrer os dados numerando-os

# mas são poucos dados então fiz manualmente

cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}

# Agora usando a função map percorro a coluna Cabin associando as letras (keys) aos valores (value) do dicionário

df_train['Cabin'] = df_train['Cabin'].map(cabin_category)
# verificando o resultado

df_train['Cabin'].unique().tolist()
# A mesma regra deve ser aplicada ao dataset de test

df_test['Cabin'] = df_test['Cabin'].map(lambda x:re.compile("([a-zA-Z])").search(x).group())

df_test['Cabin'] = df_test['Cabin'].map(cabin_category)

df_test['Cabin'].unique().tolist()
# Visualizando seus dados novamente

#df_train.head()

df_test.head()
# A coluna PassengerId, Name, Ticket a principio não são interessantes para essa primeira analise

# Obs. usando o parametro inplace = True que faz uma atualização automatica no dataframe em questão.

df_train.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)

df_test.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)
# Visualizando seus dados novamente

df_train.head()

#df_test.head()
# Uma vez que já temos somente dados numéricos vamos partir para criar o modelo...

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
# Utilizando a lib sklearn.model_selection e a função train_test_split separamos os dados em treino e teste

# você poderia primeiro criar o dataset y com os dados da coluna Survived

# e o dataset X com os demais dados sem a coluna do y, ou seja separa as dependetes das não dependentes

# mas aqui eu passei direto para dentro da função a separação dos dados

X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['Survived'], axis=1), df_train['Survived'], test_size = 0.2, random_state=2)
# Agora vamos chamar a função LogisticRegression, como nossa saída deverá ser uma classificação 0 não sobreviveu

# e 1 sobreviveu então temos um problema de classificação por isso iremos aplicar regresssão logistica, 

# embora possa ser feita poroutras tecnicas de classificação supervisionada sobre KNN.

LogisticRegression = LogisticRegression(max_iter=10000) # quantidade de rodadas

LogisticRegression.fit(X_train, y_train) # passagem dos dados de treino
# Uma vez tendo treinado os dados teremos um modelo que tentará prever os resultados do  dataset de test

# a função predict() recebe um dataset de teste para ver como o modelo esta respondendo sobre

# dados que ele ainda não conhece

predictions = LogisticRegression.predict(X_test)

predictions
# Visualizando os acertos e os erros os previstos 1 e que realmente eram um 1 = 88

# e os zeros que eram zeros = 53

conf_matrix = confusion_matrix(y_test, predictions)

print(conf_matrix)
# Podemos ver que o resultado não foi muito bom, pois 78% de certeza

# sobre a possibilidade de sobreviver ou morrer em um naufragio é bem arriscado

accuracy = ((conf_matrix[0,0] + conf_matrix[1,1]) / conf_matrix.sum()) * 100

accuracy