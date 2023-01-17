# Numpy, Pandas, Scipy

import numpy as np

import pandas as pd

from scipy.stats import norm



# Matplotlib e Sns per plotting

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Disabilità i warnings

import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

warnings.filterwarnings(action='ignore', category=FutureWarning)



# Funzioni utili

def showValues (df, var):

    print (df[var].value_counts(dropna=False))

    print ("N. di categorie:",len(df[var].value_counts().tolist()))

    print ("-------------------------")



# Lettura dati

house_train = pd.read_csv('../input/train.csv')

house_preds = pd.read_csv('../input/test.csv')
# Restituisce i nomi, utile per creare un nuovo dataset di soli object, int64 o float64

house_train.select_dtypes(include='object').columns

house_train.select_dtypes(include='int64').columns

house_train.select_dtypes(include='float64').columns.tolist()

house_train.drop("Id", axis=1, inplace=True)
house_train.fillna(0, inplace=True)

features = house_train.select_dtypes(exclude='object').columns.tolist()

print(features)

len(features) # 37



# Una semplice funzione per plottare gli istogrammi delle variabili dentro cols

# Manca SalePrice, non entra nei subplots

from scipy.stats import norm

rows = 12

cols = 3

fig, ax = plt.subplots(rows,cols)  # n.righe / n.colonne

fig.set_figheight(rows*5)          # altezza di tutto il riquadro

fig.set_figwidth(20)               # larghezza di tutto il riquadro



n = 0

for i in range(0,rows):

    for j in range(0,cols):

        sns.distplot(house_train[features[n]], fit=norm, ax=ax[i][j]) # Provare: np.sqrt, np.log1p

        n += 1
# Vediamo SalePrice

fig, ax = plt.subplots(1,1)

fig.set_figheight(5)

fig.set_figwidth(10)

sns.distplot(house_train['SalePrice'], bins=40, color="r", fit=norm, kde=True)



print("Asimmetria:",house_train['SalePrice'].skew())

print("Curtosi:",house_train['SalePrice'].kurtosis())
# Le distribuzioni asimmetriche verso destra possono essere normalizzate tramite trasformazione logaritmica

fig, ax = plt.subplots(1,1)

fig.set_figheight(5)

fig.set_figwidth(10)

sns.distplot(np.log1p(house_train['SalePrice']), bins=40, color="g", fit=norm, kde=True)



print("Asimmetria dopo log:",np.log1p(house_train['SalePrice']).skew())

print("Curtosi dopo log:",np.log1p(house_train['SalePrice']).kurtosis())
# LotArea ha una curtosi altissima dovuta ai valori verso destra

# E' un segnale di possibili valori anomali

fig, ax = plt.subplots(1,1)

fig.set_figheight(5)

fig.set_figwidth(10)

sns.distplot(house_train['LotArea'], bins=100, color="r", fit=norm, kde=True)



print("Asimmetria:",house_train['LotArea'].skew())

print("Curtosi:",house_train['LotArea'].kurtosis())
# Diamo un'occhiata alle case con LotArea > 100000

house_train.loc[house_train['LotArea'] > 100000]
# House_preds non ha case con LotArea maggiore di 100000 da predirre

# Dato che non ci servono possiamo rimuoverle dal train set

house_train = house_train[house_train['LotArea'] < 100000]
# Trasformiamo

fig, ax = plt.subplots(1,1)

fig.set_figheight(5)

fig.set_figwidth(10)

sns.distplot(np.log1p(house_train['LotArea']), bins=60, color="g", fit=norm, kde=True)

# La curva di LotArea è migliorata anche grazie alla rimozione dei valori anomali
df = pd.DataFrame(columns=['Feature', 'Skew', 'Kurtosis'])

columns = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']

for col in columns:

    df.loc[df.shape[0]] = [col, house_train[col].skew(), house_train[col].kurtosis()]

df
# Come appaiono

fig, ax = plt.subplots(1,3)

fig.set_figheight(5)

fig.set_figwidth(20)

sns.distplot(house_train['LotFrontage'], fit=norm, ax=ax[0])

sns.distplot(house_train['BsmtUnfSF'], fit=norm, ax=ax[1])

sns.distplot(house_train['2ndFlrSF'], fit=norm, ax=ax[2])
# Effettivamente la trasformazione non va bene, dunque evitiamo

fig, ax = plt.subplots(1,3)

fig.set_figheight(5)

fig.set_figwidth(20)

sns.distplot(np.log1p(house_train['LotFrontage']), color='r', fit=norm, ax=ax[0])

sns.distplot(np.log1p(house_train['BsmtUnfSF']), color='r', fit=norm, ax=ax[1])

sns.distplot(np.log1p(house_train['2ndFlrSF']), color='r', fit=norm, ax=ax[2])
df.loc[df['Skew'] > 1]
# Vediamo le altre rimanenti

fig, ax = plt.subplots(2,3)

fig.set_figheight(10)

fig.set_figwidth(20)

sns.distplot(house_train['MasVnrArea'], fit=norm, ax=ax[0][0])

sns.distplot(house_train['BsmtFinSF1'], fit=norm, ax=ax[0][1])

sns.distplot(house_train['TotalBsmtSF'], fit=norm, ax=ax[0][2])

sns.distplot(house_train['1stFlrSF'], fit=norm, ax=ax[1][0])

sns.distplot(house_train['GrLivArea'], fit=norm, ax=ax[1][1])

# sns.distplot(np.log1p(house_train['2ndFlrSF']), fit=norm, ax=ax[1][2])
# TotalBsmtSF, 1stFlrSF e GrLivArea hanno le migliori forme, comunque trasformiamo tutto per prova

fig, ax = plt.subplots(2,3)

fig.set_figheight(10)

fig.set_figwidth(20)

sns.distplot(np.log1p(house_train['MasVnrArea']), color="r", fit=norm, ax=ax[0][0])

sns.distplot(np.log1p(house_train['BsmtFinSF1']), color="r", fit=norm, ax=ax[0][1])

sns.distplot(np.log1p(house_train['TotalBsmtSF']), color="y", fit=norm, ax=ax[0][2])

sns.distplot(np.log1p(house_train['1stFlrSF']), color="g", fit=norm, ax=ax[1][0])

sns.distplot(np.log1p(house_train['GrLivArea']), color="g", fit=norm, ax=ax[1][1])

# sns.distplot(np.log1p(house_train['2ndFlrSF']), fit=norm, ax=ax[1][2])
# TotalBsmtSF non esce bene a causa dei zero a sinistra, sono corretti o anomali?

fig, ax = plt.subplots(1,1)

fig.set_figheight(5)

fig.set_figwidth(10)

sns.distplot(house_train['TotalBsmtSF'], bins=100, color="b", fit=norm, kde=True)
house_train.loc[house_train['TotalBsmtSF'] > 0].sort_values(by="TotalBsmtSF").head()
house_train.loc[house_train['TotalBsmtSF'] > 0].sort_values(by="TotalBsmtSF").tail()
house_train.loc[house_train['TotalBsmtSF'] == 0].sort_values(by="TotalBsmtSF").head()
# Su preds abbiamo case con TotalBsmtSF maggiore di 3500?

house_preds.loc[house_preds['TotalBsmtSF'] > 3500]
# Proviamo a visualizzare senza i valori maggiori di 5100, così da includere l'unica casa di preds

# Non è però detto che rimuovere queste case migliorerà il modello

df = house_train.loc[house_train['TotalBsmtSF'] < 5100]

fig, ax = plt.subplots(1,1)

fig.set_figheight(5)

fig.set_figwidth(10)

sns.distplot(df['TotalBsmtSF'], color="b", fit=norm, kde=True)