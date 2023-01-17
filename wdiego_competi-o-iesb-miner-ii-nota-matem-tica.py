# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Carregando a base de treino

df = pd.read_csv('../input/train.csv')

df.shape
# Carregando a base de teste

df_test = pd.read_csv('../input/test.csv')

df_test.shape
# Juntando as bases de treino e teste

df = df.append(df_test, ignore_index=True, sort=False)

df.shape
df.sample(5).T
df.info()
# Corrigindo tipos

df['populacao'] = df['populacao'].str.replace('.', '')

df['populacao'] = df['populacao'].str.replace('\(\)', '')

df['populacao'] = df['populacao'].str.replace('\(1\)', '')

df['populacao'] = df['populacao'].str.replace('\(2\)', '')

df['populacao'] = df['populacao'].str.replace(',', '').astype('float')

df['densidade_dem'] = df['densidade_dem'].str.replace(',', '').astype('float')

df['area'] = df['area'].str.replace(',', '').astype('float')

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('#DIV/0!', '-1')

df['comissionados_por_servidor'] = df['comissionados_por_servidor'].str.replace('%', '').astype('float')
# Tratando valores nulos

df['in_na_densidade_dem'] = df['densidade_dem'].isna().astype('int')

df['densidade_dem'].fillna(df['densidade_dem'].mean(), inplace=True)



df['in_na_participacao_transf_receita'] = df['participacao_transf_receita'].isna().astype('int')

df['participacao_transf_receita'].fillna(-1, inplace=True)



df['in_na_servidores'] = df['servidores'].isna().astype('int')

df['servidores'].fillna(-1, inplace=True)



df['in_na_perc_pop_econ_ativa'] = df['perc_pop_econ_ativa'].isna().astype('int')

df['perc_pop_econ_ativa'].fillna(-1, inplace=True)



df['in_na_gasto_pc_saude'] = df['gasto_pc_saude'].isna().astype('int')

df['gasto_pc_saude'].fillna(-1, inplace=True)



df['in_na_hab_p_medico'] = df['hab_p_medico'].isna().astype('int')

df['hab_p_medico'].fillna(-1, inplace=True)



df['in_na_exp_vida'] = df['exp_vida'].isna().astype('int')

df['exp_vida'].fillna(df['exp_vida'].mean(), inplace=True)



df['in_na_gasto_pc_educacao'] = df['gasto_pc_educacao'].isna().astype('int')

df['gasto_pc_educacao'].fillna(-1, inplace=True)



df['in_na_exp_anos_estudo'] = df['exp_anos_estudo'].isna().astype('int')

df['exp_anos_estudo'].fillna(-1, inplace=True)
ordem_porte = ['Pequeno porte 1', 'Pequeno porte 2', 'Médio porte', 'Grande porte']

df['porte'] = pd.Categorical(df['porte'], categories=ordem_porte).codes



df['div_gasto'] = df['gasto_pc_saude'] / df['pib']

df['exp_pib'] = df['pib'] / df['exp_vida']

df['exp_pib_estudos'] = df['pib'] / df['exp_anos_estudo']
# Transformando dados categóricos em números

for col in df.columns:

    if df[col].dtype == 'object':

        df[col] = df[col].astype('category').cat.codes
df.info()
# Separando a base de teste

test = df[df['nota_mat'].isnull()]

df = df[~df['nota_mat'].isnull()]
df.shape, test.shape
df = df[df['populacao'] > 5000].copy()
# Separando em treino e validação

from sklearn.model_selection import train_test_split



train, valid = train_test_split(df, test_size=0.2, random_state=42)
train.shape, valid.shape
# Selecionar as colunas para treinamento

feats = [c for c in df.columns if c not in ['nota_mat', 'codigo_mun', 'municipio']]
# Treinamento do modelo



# Importar o RandomForest

from sklearn.ensemble import RandomForestClassifier
# Instanciar o RandomForest

rf = RandomForestClassifier(n_estimators=200, min_samples_split=5, max_depth=4, random_state=42)
# Treinar o modelo

rf.fit(train[feats], train['nota_mat'])
# Fazer as previsões

preds = rf.predict(valid[feats])
# Avaliando o desempenho do modelo

from sklearn.metrics import accuracy_score



accuracy_score(valid['nota_mat'], preds)
# Verificando os dados de validação quanto ao Churn

valid['nota_mat'].value_counts()
# Analisando a importância das caracteristicas

pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
# Prevendo os valores do teste

pred_test = rf.predict(test[feats])
r1 = pd.DataFrame({'codigo_mun': test['codigo_mun'], 'nota_mat': pred_test})

r1.to_csv('r1.csv', index=False)
# Modelo 2 - cross validation



def cv(df, test, feats, y_name, k=5):

    preds, score, fis = [], [], []

    chunk = df.shape[0] // k

    

    for i in range(k):

        if i+1 < k:

            valid = df.iloc[i*chunk: (i+1)*chunk]

            train = df.iloc[:i*chunk].append(df.iloc[(i+1)*chunk:])

        else:

            valid = df.iloc[i*chunk:]

            train = df.iloc[:i*chunk]

            

        rf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=200)

        rf.fit(train[feats], train[y_name])

        score.append(accuracy_score(valid[y_name], rf.predict(valid[feats])))

        preds.append(rf.predict(test[feats]))

        fis.append(rf.feature_importances_)

        print(i, 'OK')

    

    return score, preds, fis
score, preds, fis = cv(df, test, feats, 'nota_mat')
score
r2 = pd.DataFrame({'codigo_mun': test['codigo_mun'], 'nota_mat': preds[1]})

r2.to_csv('r2.csv', index=False)
import lightgbm as lgb
# Modelo 3 - cross validation



def cv(df, test, feats, y_name, k=5):

    preds, score, fis = [], [], []

    chunk = df.shape[0] // k

    

    for i in range(k):

        if i+1 < k:

            valid = df.iloc[i*chunk: (i+1)*chunk]

            train = df.iloc[:i*chunk].append(df.iloc[(i+1)*chunk:])

        else:

            valid = df.iloc[i*chunk:]

            train = df.iloc[:i*chunk]

            

        rf = lgb.LGBMClassifier(min_child_samples=500)

        rf.fit(train[feats], train[y_name], eval_set=[(train[feats], train['nota_mat']), (valid[feats], valid['nota_mat'])], early_stopping_rounds=50)

        score.append(accuracy_score(valid[y_name], rf.predict(valid[feats])))

        preds.append(rf.predict(test[feats]))

        fis.append(rf.feature_importances_)

        print(i, 'OK')

    

    return score, preds, fis
score, preds, fis = cv(df, test, feats, 'nota_mat')
score
r3 = pd.DataFrame({'codigo_mun': test['codigo_mun'], 'nota_mat': preds[1]})

r3.to_csv('r3.csv', index=False)