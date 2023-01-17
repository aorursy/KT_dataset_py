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
train_origin = pd.read_csv('../input/train.csv')

test_origin = pd.read_csv('../input/test.csv')
train_origin.shape, test_origin.shape
train_origin.head(1)
test_origin.head(1)
train_origin.info()
test_origin.info()
df_origin = train_origin.append(test_origin, sort=False)
df_origin.shape
df_origin.head(1)
df_origin.info()
df_origin.columns[df_origin.isna().any()].tolist()
df = df_origin.copy()
df.shape
df['nota_mat'] = np.log(df['nota_mat'])
def parse_category_codes(df):  

    for column in df.columns:

        if((df[column].dtypes == 'object') & (column != 'codigo_mun')):

            df[column] = df[column].astype('category').cat.codes

    return df
df = parse_category_codes(df)
codigos = []

for codigo in df['codigo_mun']:

    codigos.append(codigo[6:])



df['codigo_mun'] = codigos
df['codigo_mun'] = df['codigo_mun'].astype(int)
df['nota_mat'].min()
df['nota_mat'].fillna(-1, inplace=True)
df.fillna(df.median(), inplace=True)
test = df[df['nota_mat'] == -1]
df = df[df['nota_mat'] != -1]
df.shape, test.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
train, valid = train_test_split(df, random_state=42)
train.shape, valid.shape
rf = RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1)
df.columns
feats = [column for column in df.columns if column not in ['nota_mat', 'codigo_mun', 'Unnamed: 0', 

                                                           'capital', 'ranking_igm', 

                                                           'comissionados_por_servidor', 'servidores',

                                                           'hab_p_medico', 'gasto_pc_saude',

                                                           'participacao_transf_receita',

                                                           'area', 'pib_pc', 'municipio', 

                                                           'indice_governanca']]
rf.fit(train[feats], train['nota_mat'])
mean_squared_error(rf.predict(valid[feats]), valid['nota_mat'])**(0.5)
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh(figsize=(25, 10), fontsize=15)
test['nota_mat'] = np.exp(rf.predict(test[feats]))
test[['codigo_mun', 'nota_mat']].to_csv('rf.csv', index=False)