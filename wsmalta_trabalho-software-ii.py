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
df = pd.read_csv('/kaggle/input/seattle-airbnb-listings/seattle_01.csv')
df.T
df.describe()
pd.options.display.float_format = '{:.0f}'.format
df.describe()
df['price'].hist(bins=20, figsize=(10,8))
df[df['price'] < 600]['price'].hist(bins=20, figsize=(10,8))
df[df['price'] >= 600]['price'].hist(bins=20, figsize=(10,8))
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(30, 4))

sns.boxplot(x=df["price"])
plt.figure(figsize=(30, 4))

sns.boxplot(x=df[df["price"]<250]['price'])
df['price'].skew()
df['price'].kurtosis()
df['price'].describe().reset_index()
def preco(arg):

    if arg > 125:

        return '4 - Muito alto'

    elif arg > 88:

        return '3 - Alto'

    elif arg >= 65:

        return '2 - Médio'

    else:

        return '1 - Baixo'
df['custo']=df.apply(lambda arg: preco(arg['price']), axis=1)
plt.figure(figsize=(10, 6))

p=sns.countplot(data=df,x="custo")
df[df['price']==65]['price'].count()
pd.options.display.float_format = '{:.10f}'.format
df[['room_type','latitude','longitude']]
sns.pairplot(x_vars=['longitude'],y_vars=['latitude'], data=df, hue='custo', hue_order = ['1 - Baixo', '2 - Médio','3 - Alto','4 - Muito alto'], height=10)
plt.figure(figsize=(10, 6))

sns.kdeplot(df['longitude'],df['latitude'], shade=True)
sns.catplot(x="room_type", y="price", data=df);
df.isnull().sum().reset_index()
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(df[['longitude','latitude','reviews','overall_satisfaction','accommodates','bedrooms','bathrooms','price']])  
IterativeImputer(add_indicator=False, estimator=None,

                 imputation_order='ascending', initial_strategy='mean',

                 max_iter=10, max_value=None, min_value=None,

                 missing_values=np.nan, n_nearest_features=None,

                 random_state=0, sample_posterior=False, tol=0.001,

                 verbose=0)
array_imputado = imp.transform(df[['longitude','latitude','reviews','overall_satisfaction','accommodates','bedrooms','bathrooms','price']])
df_imputado=pd.DataFrame(data=array_imputado[:,:],columns=['longitude','latitude','reviews','overall_satisfaction','accommodates','bedrooms','bathrooms','price'])
df_imputado.T
df_imputado.isnull().sum().reset_index()
import statsmodels.api as sm
target = df_imputado[['price']]
predictor = df_imputado[['longitude','latitude','reviews','overall_satisfaction','accommodates','bedrooms','bathrooms']]
y=target["price"]

x=predictor
model = sm.OLS(y, x).fit()

predictions = model.predict(x) 

model.summary()
predictor = df_imputado[['reviews','overall_satisfaction','accommodates','bedrooms','bathrooms']]

y=target["price"]

x=predictor

x = sm.add_constant(x)  # adiciona uma constante à regressão

model = sm.OLS(y, x).fit()

predictions = model.predict(x) 

model.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif
predictor = df_imputado[['overall_satisfaction','accommodates','bedrooms','bathrooms']]

y=target["price"]

x=predictor

x = sm.add_constant(x)  # adiciona uma constante à regressão

model = sm.OLS(y, x).fit()

predictions = model.predict(x) 

model.summary()
x.T
vif = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif
predictor = df_imputado[['overall_satisfaction','accommodates','bedrooms','bathrooms']]

y=target["price"]

x=predictor

model = sm.OLS(y, x).fit()

predictions = model.predict(x) 

model.summary()
vif = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif
predictor = df_imputado[['accommodates','bedrooms','bathrooms']]

y=target["price"]

x=predictor

model = sm.OLS(y, x).fit()

predictions = model.predict(x) 

model.summary()