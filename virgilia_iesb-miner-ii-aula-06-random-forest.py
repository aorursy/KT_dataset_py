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
# Carregando os dados

df = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/train.csv')

test = pd.read_csv('/kaggle/input/costa-rican-household-poverty-prediction/test.csv')



df.shape, test.shape
# Juntando os dataframes

df_all = df.append(test)



df_all.shape
# Quais colunas do dataframe são do tipo object

df_all.select_dtypes('object').head()

# Analisando os dados da coluna edjefa

df_all['edjefa'].value_counts()
# Vamos transformar 'yes' em 1 e 'no' em 0

# nas colunas edjefa e edjefe

mapeamento = {'yes':1, 'no':0}



df_all['edjefa'] = df_all['edjefa'].replace(mapeamento).astype(int)

df_all['edjefe'] = df_all['edjefe'].replace(mapeamento).astype(int)
# Quais colunas do dataframe são do tipo object

df_all.select_dtypes('object').head()

# Olhando a coluna dependency

df_all['dependency'].value_counts()
# Vamos transformar 'yes' em 1 e 'no' em 0

# nas colunas dependency

df_all['dependency'] = df_all['dependency'].replace(mapeamento).astype(float)
# Quais colunas do dataframe são do tipo object

df_all.select_dtypes('object').head()
# Verificando valores nulos

df_all.isnull()
# Preenchendo valores nulos com -1

df_all.fillna(-1, inplace=True)
# Separando os dataframes

train, test = df_all[df_all['Target'] != -1], df_all[df_all['Target'] == -1]



train.shape, test.shape
# Olhando os valores de Target

train['Target'].value_counts(normalize=True)
# Instanciando o RandomForest Classifier

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_jobs=-1, oob_score=True, n_estimators=200, random_state=42)
# Separar as colunar para usar no treino

feats = [c for c in train.columns if c not in ['Id', 'idhogar', 'Target']]
# Treinar o modelo

rf.fit(train[feats], train['Target'])
# Fazendo as previsões usando o modelo

test['Target'] = rf.predict(test[feats]).astype(int)
# Verificando os resultados das previsões

test['Target'].value_counts(normalize=True)
# Gerando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)
# Limitando o treinamento do chefe da família



# Criando um novo dataframe para treinar

heads = train[train['parentesco1'] == 1]
# Criando um novo modelo

rf2 = RandomForestClassifier(n_jobs=-1, n_estimators=200, oob_score=True, random_state=42)
# Treinando o modelo

rf2.fit(heads[feats], heads['Target'])
heads.shape
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf2.predict(test[feats].astype(int))
# Gerando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)
# Feature Engineering



# Vamos criar colunas para valores percapita



df_all['hsize-pc'] = df_all['hhsize'] / df_all['tamviv']

df_all['phone-pc'] = df_all['qmobilephone'] / df_all['tamviv']

df_all['tablets-pc'] = df_all['v18q1'] / df_all['tamviv']

df_all['rooms-pc'] = df_all['rooms'] / df_all['tamviv']

df_all['rent-pc'] = df_all['v2a1'] / df_all['tamviv']
# Separando o dataframe em train e test



train, test = df_all[df_all['Target'] != -1], df_all[df_all['Target'] == -1]
# Separar as colunar para usar no treino

feats = [c for c in df_all.columns if c not in ['Id', 'idhogar', 'Target']]
# Criando um novo modelo

rf3 = RandomForestClassifier(n_jobs=-1, n_estimators=200, oob_score=True, random_state=42)
# Treinando o modelo

rf3.fit(train[feats], train['Target'])
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf3.predict(test[feats].astype(int))
# Gerando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)
# Feature Importance

pd.Series(rf3.feature_importances_, index=feats).sort_values().plot.barh(figsize = (20,40))
# Copiando do campeão



rf4 = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=700,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,

                            verbose=0, class_weight='balanced')
# Treinando o modelo

rf4.fit(train[feats], train['Target'])
# Prever o Target de teste usando o modelo treinado

test['Target'] = rf4.predict(test[feats].astype(int))
# Gerando o arquivo para submissão

test[['Id', 'Target']].to_csv('submission.csv', index=False)