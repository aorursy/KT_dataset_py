# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import seaborn as sns



from pandas import Series, DataFrame

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



print(os.listdir('../input'))



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/exportarMovimentos.csv')

#df.head()
df.head(5)
dados = df

df_defaul = df

#dados["DT_MOVIMENTO"] = pd.to_datetime(dados.DT_MOVIMENTO,format='%dd/%mm/%YYYY')

df[['CD_CONTRATO','DT_MOVIMENTO','VL_MOVIMENTO']][:5]

program_types = df.groupby('CD_MOEDA')['CD_CONTRATO'].count()

plt.figure(figsize = (12, 8))

plt.title("Transações de Moedas Utilizadas nos Lançamentos")

plt.ylabel("Count")

plt.xlabel("Contratos")

sns.barplot(x = program_types.index, y = program_types.values)

site_classes = df.groupby('CD_SITUACAO')['CD_CONTRATO'].count()

plt.figure(figsize = (12, 8))

plt.title("Status dos Contratos")

plt.ylabel("Count")

plt.xlabel("Contratos Status")

sns.barplot(x = site_classes.index, y = site_classes.values)
completed_sites = df.loc[df['CD_SITUACAO'] == 'G']



completed_sites['DT_MOVIMENTO'] = pd.to_datetime(completed_sites['DT_MOVIMENTO']).dt.strftime("%m")

completed_sites.head()

completed_sites = completed_sites.groupby('DT_MOVIMENTO')['CD_CONTRATO'].count()

plt.figure(figsize = (12, 8))

plt.title("Contatos por mês Gerados")

plt.ylabel("Count")

plt.xlabel("Date")

plt.xticks(rotation = 90)

sns.lineplot(x = completed_sites.index, y = completed_sites.values)
control_types = df.groupby('CD_MOEDA')['DT_FINANCEIRA'].count()

labels = [index + ": {:.2f}%".format(count/df['CD_MOEDA'].count()*100) for index, count in zip(control_types.index, control_types.values)]

plt.figure(figsize = (15, 10))

plt.title("Percentual de Moedas nos Lançamentos")

plt.yticks(rotation = 90)

plt.pie(x = control_types.values, labels = labels)


dados = df.groupby('CD_CONTRATO').aggregate(np.sum)

dados[dados['VL_MOVIMENTO'] > 1000].plot(kind = 'bar')

# Selecionando 5 ocorrencias randomicamente

#df.sample(5,random_state=89)

df.isnull().mean().round(4)*100
import plotly

import seaborn as sns; sns.set(style='white')

df['CD_CONTRATO'].value_counts()[0:10].plot(kind='bar',title='Top 10 Contratos',fontsize=14,color='#9370DB')
df.describe().transpose()
from wordcloud import WordCloud, STOPWORDS



%matplotlib inline

text = df['CD_MOEDA'].values

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = '#F0F0F0',

    stopwords = STOPWORDS).generate(str(text))



fig = plt.figure(

    figsize = (15, 20),

    facecolor = '#F0F0F0',

    edgecolor = '#F0F0F0')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
from datetime import datetime

df["DATE"] = pd.to_datetime(df["DT_FINANCEIRA"]) # seting the column as pandas datetime

df["YEAR"] = df['DATE'].dt.year # extracting year

df["MONTH"] = df["DATE"].dt.month # extracting month

df["WEEKDAY_NAME"] = df["DATE"].dt.weekday_name # extracting name of the weekday

df["YEAR_MONTH"] = df.YEAR.astype(str).str.cat(df.MONTH.astype(str), sep='-')
df["WEEKDAY_NAME"].value_counts().plot(kind='barh',title='Data Financeira Semanalmente',color='cornflowerblue')
X = df[['YEAR','VL_MOVIMENTO']]

X.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( X, df.VL_CAIXA, test_size=0.3)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
from sklearn.linear_model import LinearRegression



lreg = LinearRegression()
lreg.fit(x_train,y_train)
pred = lreg.predict(x_test)
mse = np.mean((pred - y_test)**2)

mse
coeff = DataFrame(x_train.columns)

coeff['Coeficientes'] = Series(lreg.coef_)

coeff
lreg.score(x_test,y_test)
df.TX_JUROS.head()
df.TX_JUROS.isnull().sum()
df.TX_JUROS.fillna((df.TX_JUROS.mean()), inplace=True)
df.TX_JUROS.head()
X = df.loc[:,['YEAR','VL_MOVIMENTO','TX_JUROS']]

X.head()
x_train, x_test, y_train, y_test = train_test_split( X, df.VL_CAIXA, test_size=0.3)
lreg.fit(x_train,y_train)
pred = lreg.predict(x_test)

mse = np.mean((pred - y_test)**2)

mse
coeff = DataFrame(x_train.columns)

coeff['Coeficientes'] = Series(lreg.coef_)

coeff
lreg.score(x_test,y_test)
df['TX_JUROS'] = df['TX_JUROS'].replace(0,np.mean(df['TX_JUROS']))

df['YEAR'] = 2012 - df['YEAR']

df['QT_COT_CAIXA'].fillna(1,inplace=True)
df.CD_MOEDA.value_counts()
df.CD_SITUACAO.value_counts()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
# BK

df2 = df


#df['CD_MUTUARIO'].fillna(1,inplace=True)

df.drop(['CD_MUTUARIO','CD_EMPRESA_MOV'], axis=1,inplace=True)

#df['CD_MUTUARIO'].head()

#df['CD_MUTUARIO'] = pd.to_numeric(df['CD_MUTUARIO'])

dados.head()
def generate_labelencoder(atts):

  for attr in atts:

    df[attr] = le.fit_transform(df[attr])

  return df

# transformar colunas string em number

#df.drop(['TP_MOVIMENTO'], axis=1,inplace=True)

df = generate_labelencoder(['CD_SITUACAO','CD_MOEDA','DT_MOVIMENTO','DT_ALTERACAO','FL_REAL_PREVISTO','DT_FINANCEIRA','DATE','WEEKDAY_NAME','YEAR_MONTH'])

df.head()

#dados.head()
nan_cols = [i for i in df.columns if df[i].isnull().any()]

nan_cols

#df[nan_cols]

#df[nan_cols].fillna((df['CD_MOEDA'].mean()), inplace=True)

#df[nan_cols].fillna((1), inplace=True)

#df['CD_USUARIO_ALTERACAO'].fillna((1), inplace=True)

df = df.groupby(df.columns, axis = 1).transform(lambda x: x.fillna((1)))

df[nan_cols].head(5)

#df['TX_JUROS'].fillna(0.1,inplace=True)

#X = df[df.isnull() != 'NaN']

X = df

X.head()

X.isnull().any()
x_train, x_test, y_train, y_test = train_test_split( X, df.VL_MOVIMENTO, test_size=0.3)

x_train.dtypes.sample(18)
lreg = LinearRegression()

lreg.fit(x_train,y_train)
pred_cv = lreg.predict(x_test)
mse = np.mean((pred_cv - y_test)**2)

mse
lreg.score(x_test,y_test)
predictors = x_train.columns



coef = Series(lreg.coef_,predictors).sort_values()



coef.plot(kind='bar', title='Modal Coefficients') 
from sklearn.linear_model import Ridge

ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(x_train,y_train)
pred = ridgeReg.predict(x_test)



mse = np.mean((pred - y_test)**2)



mse
ridgeReg.score(x_test,y_test)
def plot_coeficientes_ridge(alpha):

  ridgeReg = Ridge(alpha=alpha, normalize=True)

  

  ridgeReg.fit(x_train,y_train)

  

  predictors = x_train.columns

  

  coef = Series(ridgeReg.coef_,predictors).sort_values()

  

  print(coef)

  

  coef.plot(kind='bar', title='Ridge Coefficients')

plot_coeficientes_ridge(0.01)
plot_coeficientes_ridge(0.5)
!ll