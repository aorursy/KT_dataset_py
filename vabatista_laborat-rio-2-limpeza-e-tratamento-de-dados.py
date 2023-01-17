%matplotlib inline

import pandas as pd

import numpy as np

import pylab as plt



# definições para gráficos

plt.rc('figure', figsize=(10, 5))

fizsize_with_subplots = (10, 10)

bin_size = 20



# df_train é o nosso dataframe com os dados de treinamento para construção de nosso modelo.

df_train = pd.read_csv('../input/lab2_train.csv')

df_train.head()
## Lista as colunas e marca as que possuem algum valor nulo

print(df_train.isnull().any())

print()



## Na informação básica do dataframe, podemos ver o número de valores não nulos de cada coluna.

print(df_train.info())
## Cria um novo dataframe (df_train2) sem a coluna Cabin

df_train2 = df_train.drop('Cabin', axis=1)

df_train2.info()
media_idade = df_train2['Age'].mean()

mediana_idade = df_train2['Age'].median()



print(media_idade)

print(mediana_idade)



df_train2['Age'].hist()

plt.title('Idade')
## preenche os nulos com a média das idades

df_train2['Age'].fillna(media_idade, inplace=True) ## O inplace altera no próprio dataframe em vez de termos de criar outro

df_train2['Age'].hist()

plt.title('Idade')
# histograma só é possível com valores numéricos, mas Embarked é categórico

df_train2['Embarked'].value_counts().plot(kind='bar', title='Porto de embarque') 
df_train2['Embarked'].fillna('S', inplace=True)

df_train2.info()
df_train2.to_csv('lab2_train_no_nulls.csv', index=False)
# Vamos carregar nossos dados já sem os nulos

df_train = pd.read_csv('lab2_train_no_nulls.csv')
df_train.describe()
print(df_train.sort_values('Age', ascending=False).head(5)['Age'])

print(df_train.sort_values('Age', ascending=True).head(5)['Age'])
df_train.loc[df_train['Age']==133, 'Age'] = media_idade ## Aqui substituimos os valores de idade iguais a 133 pela média
print(df_train.sort_values('Fare', ascending=False).head(5)['Fare'])

print(df_train.sort_values('Fare', ascending=True).head(5)['Fare'])
mediana_tarifa = df_train['Fare'].median()

df_train.loc[df_train['Fare']>5000, 'Fare'] = mediana_tarifa

df_train.loc[df_train['Fare']<0, 'Fare'] = mediana_tarifa
print(df_train.sort_values('Fare', ascending=False).head(5)['Fare'])

print(df_train.sort_values('Fare', ascending=True).head(5)['Fare'])
df_train.to_csv('train_no_nulls_no_outliers.csv', index=False)
##Recarregando os dados da parte anterior...

df_train = pd.read_csv('train_no_nulls_no_outliers.csv')

df_train.head(2)
novas_colunas = pd.get_dummies(df_train['Embarked']) 

df_train2 = pd.concat([df_train,novas_colunas], axis=1) # axis = 1 concatena colunas. axis = 0 concatena linhas

df_train2.head(3)
df_train2.drop('Embarked', axis=1, inplace=True)
novas_colunas_pclass = pd.get_dummies(df_train['Pclass']) 

novas_colunas_sex = pd.get_dummies(df_train['Sex']) 



df_train3 = pd.concat([df_train2,novas_colunas_pclass, novas_colunas_sex], axis=1)

df_train3.drop(['Pclass', 'Sex'], axis=1, inplace=True)

df_train3.head(3)
df_train3.to_csv('train_no_nulls_no_outliers_ohe.csv', index=False)
##Recarregando os dados da parte anterior, antes de fazer o OHE

df_train = pd.read_csv('train_no_nulls_no_outliers.csv')

df_train.head(2)
import hashlib



def hashFunction(numero_colunas, dict_linha):

    novas_features = [0]*numero_colunas # cria uma lista vazia com o tamanho pré-determinado

    

    for coluna in dict_linha:

        coluna_e_valor = (str(coluna) + str(dict_linha[coluna])).encode('utf-8')

        posicao = int(int(hashlib.md5(coluna_e_valor).hexdigest(), 16) % numero_colunas) # calcula o hash

        novas_features[posicao] += 1 # adiciona 1 na posição onde caiu o hash



    return novas_features
coluna_e_valor = 'Animal_Rato'.encode('utf-8')

print(int(hashlib.md5(coluna_e_valor).hexdigest(), 16))



#int(int(hashlib.md5(coluna_e_valor).hexdigest(), 16) % numero_colunas)
dict_categorias = df_train[['Sex', 'Embarked', 'Pclass']].T.to_dict()

print(dict_categorias.get(0))

print(dict_categorias.get(1))
num_colunas = 4

for i in range(num_colunas):

    print(hashFunction(num_colunas, dict_categorias[i]))
num_colunas = 5



novas_colunas = []

for dict_ in dict_categorias.values():

    novas_colunas.append(hashFunction(num_colunas, dict_))

    

# convertemos nossa lista em dataframe nomeando as colunas como h0, h1, h2 ... hn

novas_colunas = pd.DataFrame(novas_colunas, columns=['h'+str(i) for i in range(num_colunas)]) 

novas_colunas.head()
df_train2 = pd.concat([df_train,novas_colunas], axis=1)

df_train2.head()
df_train2.to_csv('train_no_nulls_no_outliers_feat_hash.csv', index=False)
## recarregando nossos dados

df_train = pd.read_csv('train_no_nulls_no_outliers_feat_hash.csv')

df_train.head(2)
from sklearn import preprocessing



## dados originais

age_original = df_train['Age'].values.reshape(-1, 1)

## Normaliza os dados

age_standard = preprocessing.StandardScaler().fit_transform(df_train['Age'].values.reshape(-1, 1)) 

## Muda a escala dos dados para valores entre 0 e 1 (valores padrão, que poderiam ser personalizados)

age_minmax = preprocessing.MinMaxScaler().fit_transform(df_train['Age'].values.reshape(-1, 1))
from matplotlib import pyplot as plt



def plot():

    plt.figure(figsize=(8,6))



    plt.scatter([0]*len(age_original), age_original,

            color='green', label='Original', alpha=0.5)



    plt.scatter([1]*len(age_original), age_standard, color='red',

            label='Normalizado', alpha=0.3)



    plt.scatter([2]*len(age_original), age_minmax,

            color='blue', label='escala entre [min=0, max=1]', alpha=0.3)



    plt.xlabel('Idade')

    plt.ylabel('Idade')

    plt.legend(loc='upper left')

    plt.grid()



    plt.tight_layout()



plot()

plt.show()