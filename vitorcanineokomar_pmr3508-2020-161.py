import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
train = pd.read_csv('../input/adult-pmr3508/train_data.csv', index_col='Id')

test = pd.read_csv('../input/adult-pmr3508/test_data.csv', index_col='Id')
train.head()
plt.figure(figsize=(15,8))

g = sns.countplot(train['native.country'])

g.set(xticklabels=[])

g.tick_params(bottom=False)

sns.despine(bottom=True, left=True)

plt.show()
plt.figure(figsize=(12,6))

sns.distplot(train['capital.gain'], kde=False)

sns.despine(bottom=True, left=True)
plt.figure(figsize=(16,4))

sns.boxplot(train['capital.gain'])

sns.despine(bottom=True, left=True)
plt.figure(figsize=(12,6))

sns.distplot(train['capital.loss'], kde=False)

sns.despine(bottom=True, left=True)
plt.figure(figsize=(16,4))

sns.boxplot(train['capital.loss'])

sns.despine(bottom=True, left=True)
print('Assimetria de capital.gain é {}'.format(train['capital.gain'].skew()))

print('Assimetria de capital.loss é {}'.format(train['capital.loss'].skew()))
train['income'] = train['income'].map({'<=50K': 0, '>50K': 1})

del train['fnlwgt']
#aplicando as mesmas mudanças ao dados de teste

del test['fnlwgt']
train.isnull().any()
train['workclass'].unique()
for col in train.columns:

    if '?' in train[col].unique():

        print(" '?' presente na coluna {}".format(col))
n_lines = train.shape[0] #numero de linhas do dataset
colunas_com_problemas = ['workclass', 'occupation', 'native.country']
for col in colunas_com_problemas:

    n = (train[col]=='?').sum() #numero de vezes que '?' aparece na coluna

    prop = n/n_lines #proporcao do caracter em relacao ao todo

    print("Na coluna {} há {:.2f}% de '?'".format(col, prop*100)) 

    
train.replace('?', np.nan, inplace=True)
for col in colunas_com_problemas:

    train[col] = train[col].fillna(train[col].mode()[0])

    assert train[col].isnull().any() == False
#aplicando as mesmas mudanças ao dataset de teste

test.replace('?', np.nan, inplace=True)

for col in colunas_com_problemas:

    test[col] = test[col].fillna(test[col].mode()[0])

    assert test[col].isnull().any() == False
del train['education']

del test['education']
plt.figure(figsize=(16,4))

sns.boxplot(np.log(train['capital.gain']+1))

sns.despine(bottom=True, left=True)
train['capital.gain'] = np.log(train['capital.gain']+1)

train['capital.loss'] = np.log(train['capital.loss']+1)

test['capital.gain'] = np.log(test['capital.gain']+1)

test['capital.loss'] = np.log(test['capital.loss']+1)

#o +1 é usado para evitar de haver log(0) 
features_categoricas = list(train.select_dtypes(include='object'))

features_categoricas
features_numericas = list(train.select_dtypes(include='number'))

features_numericas.remove('income') #removendo a target das features

features_numericas
dummies = pd.get_dummies(train[features_categoricas])
train_prep = train[features_numericas]

train_prep = pd.merge(train_prep, dummies, left_index=True, right_index=True)

train_prep
#aplicando mesmas mudanças ao set de testes

dummies = pd.get_dummies(test[features_categoricas])

test_prep = test[features_numericas]

test_prep = pd.merge(test_prep, dummies, left_index=True, right_index=True)
diff = set(train_prep.columns) - set(test_prep.columns)

diff
diff.issubset( set(train_prep.columns) )
test_prep['native.country_Holand-Netherlands'] = np.zeros((test_prep.shape[0], 1))

assert diff.issubset( set(test_prep.columns))
for col in features_numericas:

    mean = train_prep[col].mean()

    sigma = train_prep[col].std()

    train_prep[col] = (train_prep[col] - mean)/sigma

    test_prep[col] = (test_prep[col]-mean)/sigma

    #teste foi normalizado com a mesma media e desvio padrao do set de treino

    #isso faz com que ambos estejam na mesma escala
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
y = train['income'] #target
model = KNeighborsClassifier()
scores = cross_val_score(model, train_prep, y, scoring='accuracy',cv=5)

scores.mean()
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [i for i in range(1, 31)],

             'p': [1, 2]}

model = KNeighborsClassifier()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(train_prep, y)
grid_search.best_params_
model = KNeighborsClassifier(n_neighbors=28, p=1)

scores = cross_val_score(model, train_prep, y, scoring='accuracy',cv=5)

scores.mean()
model.fit(train_prep, y)

pred = model.predict(test_prep)
pred = pd.Series(pred).map({0: '<=50K',1: '>50K'})
pred = pd.DataFrame(pred, columns=['income'])
pred.index.names = ['Id']
pred.to_csv('PMR-3508-2020-161.csv')