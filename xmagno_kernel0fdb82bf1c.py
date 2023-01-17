# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn import preprocessing	

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression



import statsmodels.api as sm

import seaborn as sns

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/lotofacil_Atual.csv')

dados=df

le = preprocessing.LabelEncoder()

dados ["cat9"] = le.fit (dados['padrao9']) .transform (dados['padrao9'])

dados ["cat5"] = le.fit (dados['padrao5']) .transform (dados['padrao5'])

dados.head()
dados[['par','cat5','padrao5','cat9','padrao9','atualXanterior','qtdMaiSai']].describe()
dados[dados.columns[1:16]].plot.density()

dados[dados.columns[1:16]].plot.box ()
sns.lmplot(x='D1',y='cat9',data=dados,fit_reg=True)

sns.lmplot(x='cat5',y='D1',data=dados,fit_reg=True)
dados.groupby(['padrao9','padrao5'])['par'].count().head()

dados[(dados.padrao9 == 69) & (dados.padrao5 == 267)].par.value_counts().head()
padrao9 = dados.groupby(['par','padrao9']).par.count()

padrao9.unstack().plot.bar()
padrao5 = dados.groupby(['par','padrao5']).par.count()

#padrao5 = padrao5.sort_values(ascending=False)

#padrao5.values.sort()

padrao5.unstack().plot.bar()
dados[(dados.par == 7 ) & (dados.padrao5 == 555 )].par.value_counts().head()

padrao5[75:100].unstack().plot.bar()
import ast

from sklearn.metrics import r2_score



dezenas=dados.dezenas.tolist()



def applyer(row):

    row = ast.literal_eval(row)

    if dezenas.index(str(row)) == 0: return 0

    return r2_score(row,ast.literal_eval(dezenas[dezenas.index(str(row)) - 1]))





temp = dados['dezenas'].apply(applyer)

dados['r2_score'] = temp

dados[['par','padrao9','r2_score']].head(20)

from sklearn.preprocessing import Normalizer,OneHotEncoder, LabelEncoder



enc = OneHotEncoder()

#pares = pd.get_dummies(dados[['par']])

pares = DataFrame(enc.fit_transform(dados[['par']]).toarray())

#sns.pairplot(dados[['par','padrao9','atualXanterior']])

pd.scatter_matrix(dados[['par','padrao9','atualXanterior']], alpha=0.2, figsize=(10, 10))

plt.show()
temp=dados['dezenas'].apply(lambda x: np.mean(ast.literal_eval(x)))

dados['media_dezenas'] = temp

dados.head(10)

                            