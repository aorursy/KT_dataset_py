import numpy as np

import pandas as pd

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

import matplotlib

from sklearn.metrics import accuracy_score

df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
df['nota_mat'] = np.log(df['nota_mat'])
df = df.append(test, sort=False)
df.shape
for c in df.columns:

    if (df[c].dtype == 'object') & (c != 'codigo_mun'):

        df[c] = df[c].astype('category').cat.codes
#Retirando a string ID_ID_ da Coluna

df['codigo_mun'] = df['codigo_mun'].str.replace('ID_ID_','')
# -2 para a separação posterior das bases

df['nota_mat'].fillna(-2, inplace=True)
df.fillna(0, inplace=True)
df, test = df[df['nota_mat']!=-2], df[df['nota_mat']==-2]
train, valid = train_test_split(df, random_state=42)
rf = RandomForestRegressor(random_state=42, n_estimators=100)
feats = [c for c in df.columns if c not in ['nota_mat']]
rf.fit(train[feats], train['nota_mat'])
valid_preds = rf.predict(valid[feats])
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
mean_squared_error(valid['nota_mat'], valid_preds)**(1/2)
test['nota_mat'] = np.exp(rf.predict(test[feats]))
test[['codigo_mun','nota_mat']].to_csv('rf1.csv', index=False)