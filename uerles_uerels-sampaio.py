import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import colorsys

plt.style.use('seaborn-talk')

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

import seaborn as sns



import scipy.stats as stats

import sklearn 

import math
import pandas as pd

import numpy as np

import statsmodels.api as sm

import scipy.stats as st

import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.metrics import confusion_matrix

import matplotlib.mlab as mlab

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.columns
train.head(5)
test.head(5)
train = train.drop(['regiao', 'estado', 'municipio','capital','porte', 'area','densidade_dem',

                   'servidores','comissionados','comissionados_por_servidor','taxa_empreendedorismo',

                   'jornada_trabalho','ranking_igm','anos_estudo_empreendedor'], axis=1)
test = test.drop(['regiao', 'estado', 'municipio','capital','porte', 'area','densidade_dem',

                   'servidores','comissionados','comissionados_por_servidor','taxa_empreendedorismo',

                   'jornada_trabalho','ranking_igm','anos_estudo_empreendedor'], axis=1)
df1=train
df2=test
df1.info()
df2.info()
df1.head(5)
df2.head(5)
for c in df1.columns:

    if df1[c].dtype == 'object':

        df1[c] = df1[c].astype('category').cat.codes
df1.info()
for c in df2.columns:

    if df2[c].dtype == 'object':

        df2[c] = df2[c].astype('category').cat.codes
df2.info()
df1.info()
df1.isnull().sum().sort_values(ascending=False).head(10)
df1['gasto_pc_educacao'].fillna(df1['gasto_pc_educacao'].mean(), inplace=True)

df2['gasto_pc_educacao'].fillna(df2['gasto_pc_educacao'].mean(), inplace=True)
df1.head(10)
df1['indice_governanca'].fillna(df1['indice_governanca'].mean(), inplace=True)

df2['indice_governanca'].fillna(df2['indice_governanca'].mean(), inplace=True)
df1['gasto_pc_saude'].fillna(df1['gasto_pc_saude'].mean(), inplace=True)

df2['gasto_pc_saude'].fillna(df2['gasto_pc_saude'].mean(), inplace=True)
df1['gasto_pc_saude'].fillna(df1['gasto_pc_saude'].mean(), inplace=True)

df2['gasto_pc_saude'].fillna(df2['gasto_pc_saude'].mean(), inplace=True)
df1['participacao_transf_receita'].fillna(df1['participacao_transf_receita'].mean(), inplace=True)

df2['participacao_transf_receita'].fillna(df2['participacao_transf_receita'].mean(), inplace=True)
df1['hab_p_medico'].fillna(df1['hab_p_medico'].mean(), inplace=True)

df2['hab_p_medico'].fillna(df2['hab_p_medico'].mean(), inplace=True)
df1['idhm'].fillna(df1['idhm'].mean(), inplace=True)

df2['idhm'].fillna(df2['idhm'].mean(), inplace=True)
df1.isnull().sum().sort_values(ascending=False).head(10)
df2.isnull().sum().sort_values(ascending=False).head(10)
df1.dropna(inplace=True)
df1.isnull().sum().sort_values(ascending=False).head(10)
df2.isnull().sum().sort_values(ascending=False).head(10)
#criando modelo
from sklearn.model_selection import train_test_split
X = df1.drop('nota_mat', axis=1)

y = df1['nota_mat']
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42, max_depth=2)
removed_cols = ['taxa_empreendedorismo', 'nota_mat']
feats = [c for c in df1.columns if c not in removed_cols]
dt.fit(train[feats], train['nota_mat'])
from sklearn.metrics import mean_squared_error
train['preds'] = train['nota_mat'].mean()
mean_squared_error(train['nota_mat'], train['preds'])
del train['preds']
sns.lmplot(x='pib_pc', y='nota_mat', data=df1)
from scipy.stats.stats import pearsonr
sns.pairplot(df1)
df1.corr() #Matriz de correlação
sns.heatmap(df1.corr(),cmap="coolwarm") #gráfico da correlação de Pearson
sns.distplot(df1['nota_mat'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from sklearn.model_selection import train_test_split
train, valid = train_test_split(df1, random_state=42)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42, max_depth=2)
dt.fit(train[feats], train['nota_mat'])
from sklearn.metrics import mean_squared_error
train['preds'] = train['nota_mat'].mean()
train.shape, train['nota_mat'].mean()
mean_squared_error(train['nota_mat'], train['preds'])
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf.fit(train[feats], train['nota_mat'])
train_preds = rf.predict(train[feats])
train_preds
mean_squared_error(train['nota_mat'], train_preds)**(1/2)
UERLES = pd.DataFrame()

UERLES ['codigo_mun'] = df2['codigo_mun']

UERLES ['nota_mat'] = lm.predict(df2)
UERLES.to_csv('UERLES.csv', index=False)