# Numpy e pandas

import numpy as np

import pandas as pd



# Matplotlib e Sns per plotting

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Disabilità i warnings

import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

warnings.filterwarnings(action='ignore', category=FutureWarning)
# Mostra cosa c'è dentro la cartella input

import os

print(os.listdir("../input"))
house_train = pd.read_csv('../input/train.csv')

house_to_predict = pd.read_csv('../input/test.csv')
house_train.head()
house_train.tail()
# Info() mostra: n. di entries (righe), n. di variabili (colonne) e il loro tipo, uso memoria

house_train.info()
# Descrive i dati mostrando per ogni variabile: conteggio, media, std, valore minimo e massimo, valore max quartile

# Nota: mostra solo le variabili numeriche, quelle categoriche non sono incluse

house_train.describe().T
# Mostra solo il nome delle colonne, anche se info() è gia sufficiente

house_train.columns
# Shape mostra quante righe (entries) e colonne (variabili)

house_train.shape
# Dtypes mostra il tipo di ogni variabile, i tipi sono:

# Object: le sue categorie sono caratteri/stringhe

# Int64 e Float64: come da definizione sono interi o numeri float

house_train.dtypes
# Mostra un dataset di sole colonne object

house_train.select_dtypes(include='object')             # mostra tutto

house_train.select_dtypes(include='object').head().T    # i primi cinque
house_train.select_dtypes(include='float64').head()    # solo tre colonne: LotFrontage, MasVnrArea, GarageYrBlt

house_train.select_dtypes(include='int64').head()
# Restituisce i nomi, utile per creare un nuovo dataset di soli object, int64 o float64

house_train.select_dtypes(include='object').columns

house_train.select_dtypes(include='int64').columns

house_train.select_dtypes(include='float64').columns.tolist()
# Solo numerici

house_train.select_dtypes(exclude='object').columns
# sort_values() ordina i valori in modo ascendente se non specificato

# unique() mostra le categorie della variabile

print ("Categorie di BedroomAbvGr:\n", house_train['BedroomAbvGr'].sort_values().unique())

print ("Categorie di Neighborhood:\n", house_train['Neighborhood'].sort_values().unique())
# value_counts() conta quanti sono i valori per ogni categoria

print (house_train['Alley'].value_counts())



# Per conoscere quante sono le categorie si può fare la seguente:

a = house_train['Alley'].value_counts().tolist()

print ("N. di categorie: ",len(a))
# Si puo creare una funzione diretta così da richiamarla in qualsiasi parte del programma:

# Dropna=False fa in modo di mostrare i valori mancanti (NaN)

def showValues (df, var):

    print (df[var].value_counts(dropna=False))

    print ("N. di categorie:",len(df[var].value_counts().tolist()))

    print ("-------------------------")

    

showValues(house_train, "Fence")

showValues(house_train, "GarageCond")

showValues(house_train, "Neighborhood")

showValues(house_train, "SaleCondition")
# Istogramma - per valori continui e range ampi

# Pro: mostra i valori "a colpo d'occhio" e la loro distribuzione

# Contro: i singoli valori non sono visibili perche inseriti in ogni bin

sns.set(style="darkgrid")

sns.distplot(house_train['SalePrice'])
from scipy.stats import norm

fig, ax = plt.subplots(1,1) # 1,x dove x è il numero di grafici da mostrare

fig.set_figheight(5)        # altezza di tutto il riquadro

fig.set_figwidth(16)        # larghezza di tutto il riquadro

sns.distplot(house_train['SalePrice'], bins=40, fit=norm)
from scipy.stats import norm

fig, ax = plt.subplots(1,2) # n.righe / n.colonne

fig.set_figheight(5)        # altezza di tutto il riquadro

fig.set_figwidth(20)        # larghezza di tutto il riquadro

sns.distplot(house_train['LotArea'], fit=norm, ax=ax[0])

sns.distplot(house_train['TotalBsmtSF'], fit=norm, ax=ax[1])
# Bar count - per valori discreti e categorici, o range di valori limitato

sns.countplot(house_train['PoolQC'])
# Per mostrare piu valori:

fig, ax = plt.subplots(1,4) # 1,x dove x è il numero di grafici da mostrare



fig.set_figheight(5) # altezza di tutto il riquadro

fig.set_figwidth(20) # larghezza di tutto il riquadro



sns.countplot(house_train['PoolQC'], ax=ax[0])

sns.countplot(house_train['MSZoning'].sort_values(ascending=True), ax=ax[1])

sns.countplot(house_train['GarageQual'], ax=ax[2])

sns.countplot(house_train['GarageCond'], ax=ax[3])

fig.show()



# Notare che ho scelto sono variabili categoriche e con un range limitato di valori
# Un altro esempio con una variabile categorica

# Abbiamo bisogno che il grafico sia piu grande e le label disposti nell'asse Y



fig, ax = plt.subplots(1,1)

fig.set_figheight(10)

fig.set_figwidth(12)

ax = sns.countplot(y="Neighborhood", data=house_train)
def showMissingData (df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data

    

missing = showMissingData(house_train)

missing.head(20)
missing = showMissingData(house_to_predict)

missing.head(35)
# Feature dove si trovano i valori nan, divisi per tipo: categorico e numerico

list_cat_nan = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageCond", 

                "GarageType", "GarageFinish", "GarageQual", "BsmtExposure", "BsmtFinType2",

                "BsmtFinType1", "BsmtCond", "BsmtQual", "MasVnrType", "Electrical", "SaleCondition", 

                "MSZoning", "Functional", "Utilities", "SaleType", "Exterior1st", "Exterior2nd", "KitchenQual"]



list_num_nan = ["LotFrontage", "GarageYrBlt", "MasVnrArea", "BsmtFullBath", "BsmtHalfBath", 

                "TotalBsmtSF", "GarageArea", "BsmtUnfSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1"]



# Oppure:

cat = [f for f in house_train.columns if house_train.dtypes[f] == 'object']

num = [f for f in house_train.columns if house_train.dtypes[f] != 'object']



# Usiamo queste due liste con un ciclo For per riempire i campi NaN con None (se categorici) e valore zero (se numerici)
# Con un ciclo For sia per il training set che per il set di kaggle:



for feature in list_cat_nan:

    house_train[feature] = house_train[feature].fillna("None")

    house_to_predict[feature] = house_to_predict[feature].fillna("None")

    

for feature in list_num_nan:

    house_train[feature] = house_train[feature].fillna(0)

    house_to_predict[feature] = house_to_predict[feature].fillna(0)

    

print ("Fatto: valori NaN riempiti")
# PoolQC: i valori mancanti sono dovuti al fatto che non c'è la piscina, mettiamo quindi "None"

house_train['PoolQC'] = house_train['PoolQC'].fillna("None")

house_to_predict['PoolQC'] = house_to_predict['PoolQC'].fillna("None")

house_train['PoolQC']



# In genere la migliore soluzione è creare una funzione per questo tipo di modifiche

# dato che vanno ripetuti anche nel file test di Kaggle
# Quando sono più colonne si passano come lista [[x, y...]]:

house_train[['FireplaceQu', 'PoolQC']] = house_train[['FireplaceQu', 'PoolQC']].fillna("None")

house_to_predict[['FireplaceQu', 'PoolQC']] = house_to_predict[['FireplaceQu', 'PoolQC']].fillna("None")



# Oppure:

'''

cat = ['FireplaceQu', 'PoolQC', '...']

house_train[cat] = ...

'''
# Iniziamo a vedere la relazione tra Neighborhood e SalePrice

# Un boxplot è la scelta migliore perche mostra il prezzo minimo reale e più dati

# Barplot non è indicato per questa relazione perchè SalePrice non parte da zero in ogni vicinato



fig, ax = plt.subplots(1,1)

fig.set_figheight(10)

fig.set_figwidth(12)



df = house_train[["Neighborhood", "SalePrice", "OverallCond", "SaleCondition", "TotalBsmtSF", "LotArea"]]

df = df.sort_values(["SalePrice"]).reset_index(drop=True)

sns.boxplot(y="Neighborhood", x="SalePrice", data=df)
# OverallCond

fig, ax = plt.subplots(1,1)

fig.set_figheight(6)

fig.set_figwidth(9)

sns.boxplot(x="OverallCond", y="SalePrice", data=df)
# Swarmplot è un boxplot che mostra anche i valori

fig, ax = plt.subplots(1,1)

fig.set_figheight(6)

fig.set_figwidth(10)

sns.swarmplot(x="SaleCondition", y="SalePrice", data=df)
# Scatterplot, da usare quando le due variabili sono continue ed entrambe hanno un range di valori ampio

# PRO: Distribuzione dei valori a colpo d'occhio, permette di vedere anche i valori anomali

# CONTRO: Con troppi valori diventa illeggibile

fig, ax = plt.subplots(1,1)

fig.set_figheight(6)

fig.set_figwidth(8)

sns.scatterplot(x="TotalBsmtSF", y="SalePrice", data=df)
# Notare i valori anomali

fig, ax = plt.subplots(1,1)

fig.set_figheight(6)

fig.set_figwidth(12)

s1=sns.scatterplot(x="LotArea", y="SalePrice", data=df)

# Per aggiungere una nota nel grafico

s1.text(215245, 390000, "wat", horizontalalignment='center', size='medium', color='black', weight='bold')
# Come si vede OverallCond (categorica) vs SalePrice (continua)

fig, ax = plt.subplots(1,1)

fig.set_figheight(6)

fig.set_figwidth(8)

sns.scatterplot(x="OverallCond", y="SalePrice", data=house_train)
# Scatter dove x e y sono uguali

fig, ax = plt.subplots(1,1)

fig.set_figheight(6)

fig.set_figwidth(8)

sns.scatterplot(x="TotalBsmtSF", y="TotalBsmtSF", data=df)
# Selezioniamo le righe con valori anomali in LotArea

df = house_train.copy()

df[df['LotArea'] > 100000].T
# Per visualizzarli usiamo il comando loc [index, nome colonna]

df.loc[249, 'LotArea']

df.loc[335, 'SalePrice']

df.loc[313, ['LotArea','SalePrice']]

df.loc[706, ['LotArea','SalePrice', 'SaleType', 'GarageYrBlt']]
df[(df['MSZoning'] == "RM")]

df[df['TotalBsmtSF'] > 6000]
# Quando vogliamo indicare più di una condizione:

df[(df['TotalBsmtSF'] > 6000) & (df['LotArea'] > 100000)] # in questo caso non ci sono risultati
df[(df['SalePrice'] > 600000)]
# Se non vogliamo correggerli, conoscendo gli indici possiamo rimuoverli

# Nota: inplace va in modo che la modifica venga fatta nello stesso dataset, altrimenti crea una copia

df.drop([249, 313, 335, 706], inplace=True)
# Boxplot può mostrare i valori anomali: sono i puntini neri

sns.set(style="whitegrid")

fig, ax = plt.subplots(1,1)

fig.set_figheight(4)

fig.set_figwidth(8)

sns.boxplot(x='SalePrice', data=df)
# Anche Scatterplot è molto utile

fig, ax = plt.subplots(1,1)

fig.set_figheight(6)

fig.set_figwidth(8)

sns.scatterplot(x="LotArea", y="LotArea", data=house_train)
# Calcolo Z-Score e IQR - Fa uso di tutti e due i metodi sotto descritti

# Consigliato: codice compatto, veloce e preciso

deviation = 3



# Colonna per colonna, ad es. BsmtFinSF1

df[np.abs(df.BsmtFinSF1-df.BsmtFinSF1.mean()) <= (deviation*df.BsmtFinSF1.std())] # esclude le righe con valori anomali

df[np.abs(df.BsmtFinSF1-df.BsmtFinSF1.mean()) > (deviation*df.BsmtFinSF1.std())] # mostra le righe con valori anomali
# Per filtrare così da non mostrare le altre colonne

print(df[['Id', 'BsmtFinSF1']][np.abs(df.BsmtFinSF1-df.BsmtFinSF1.mean()) > (deviation*df.BsmtFinSF1.std())])



fig, ax = plt.subplots(1,1)

fig.set_figheight(4)

fig.set_figwidth(8)

sns.boxplot(x="BsmtFinSF1", data=house_train)
# Z-Score: possiamo calcolare la deviazione di ogni variabile con una funzione matematica

# Nota: solo per variabili numeriche

from scipy import stats



numeric = [f for f in house_train.columns if house_train.dtypes[f] != 'object']



z = np.abs(stats.zscore(df[numeric]))

print(z)

print("\n",np.where(z > 3)) # z è la soglia di deviazioni max.

# L'array contiene la riga e la colonna identificato come valore anomalo



# Crea un dataset con le righe dove è presente almeno un valore anomalo

df[numeric][(np.abs(stats.zscore(df[numeric])) < 3).all(axis=1)]
# IQR: Interquartile Range, individuiamo i valori anomali che si trovano fuori dai quartili

# E' il metodo di calcolo dei valori anomali in boxplot

Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

# IQR di ogni colonna

IQR



# Valori che sono fuori IQR - True sono i valori anomali

(df[numeric] < (Q1 - 1.5 * IQR)) | (df[numeric] > (Q3 + 1.5 * IQR))



# Da completare mappando questo dataset di True/False con il dataset reale
# Correlation Matrix (heatmap)

# Nota: Mostra solo per valori interi e float

def generate_heatmap(data, tabular):

    data = data

    corrmat = data.corr()

    if tabular == 0:

        f, ax = plt.subplots(figsize=(30, 22))

        sns.heatmap(corrmat, vmax=.8, square=False, annot=True, center=0, linewidths=.5, ax=ax, cmap="RdBu_r") # BuGn_r, BrBG, RdBu_r

        ax.set_xticklabels(ax.get_xticklabels(), rotation=30);

    else:

        return corrmat



df = pd.DataFrame(house_train)

df = df.drop(["Id"], axis=1) # ID non ci serve

generate_heatmap(df, False)



# Si può continuare droppando le colonne che non sembrano mostrare nessuna relazione per ridurre la heatmap
# Usando dtypes, crea un dataset di sole variabili categoriche

df_cat = house_train[house_train.select_dtypes(include='object').columns]

df_cat["SalePrice"] = house_train["SalePrice"]

df_cat.head()



# Crea un altro dataset con sole variabili int e float

df_num = house_train[house_train.select_dtypes(exclude='object').columns]

df_num.head()



# Per FacetGrid: crea una lista contenente i nomi delle colonne di un tipo specifico

cat = [f for f in house_train.columns if house_train.dtypes[f] == 'object']

num = [f for f in house_train.columns if house_train.dtypes[f] != 'object']



df_cat.head()

cat
df_cat = df_cat.sort_values(["SalePrice"]).reset_index(drop=True)

sns.boxplot(x="Street", y="SalePrice", data=df_cat)
def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    # x=plt.xticks(rotation=90)

    

cat = [f for f in house_train.columns if house_train.dtypes[f] == 'object']

num = [f for f in house_train.columns if house_train.dtypes[f] != 'object']



df = house_train # Per non modificare il dataset originale

df = df.sort_values(["SalePrice"]).reset_index(drop=True)



f = pd.melt(df, id_vars=['SalePrice'], value_vars=cat)

g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=6)

g = g.map(boxplot, "value", "SalePrice")
sns.pairplot(df[num])
dummy_house_train = pd.get_dummies(house_train)

dummy_house_pred = pd.get_dummies(house_to_predict)



dummy_house_train.head()
train_stats = dummy_house_train.describe().T

train_stats
print(train_stats.loc['SalePrice']['mean'])

print(train_stats.loc['SalePrice']['std'])
# Normalizzazione con media e std

def norm(x):

    return (x - train_stats['mean']) / train_stats['std']



# La formula inversa (ci serve per SalePrice)

def normReverse(x):

    return x * train_stats.loc['SalePrice']['std'] + train_stats.loc['SalePrice']['mean']



# SalePrice prima di essere normalizzata

df = dummy_house_train.copy()



from scipy.stats import norm

fig, ax = plt.subplots(1,1) # 1,x dove x è il numero di grafici da mostrare

fig.set_figheight(5)        # altezza di tutto il riquadro

fig.set_figwidth(16)        # larghezza di tutto il riquadro

sns.distplot(df['SalePrice'], bins=40, fit=norm)
# Proviamo a normalizzare solo SalePrice

df.SalePrice = (df.SalePrice - df.SalePrice.mean()) / df.SalePrice.std()



# Come appare adesso

fig, ax = plt.subplots(1,1) # 1,x dove x è il numero di grafici da mostrare

fig.set_figheight(5)        # altezza di tutto il riquadro

fig.set_figwidth(16)        # larghezza di tutto il riquadro

sns.distplot(df['SalePrice'], bins=40, fit=norm)



# Si può notare che la forma non cambia ma il centro adesso è posto sul valore zero
# Ritorniamo indietro

df.SalePrice = normReverse(df.SalePrice)



# Proviamo la trasformazione logaritmica

# In questo caso è la migliore perche i dati sono spostati verso sinistra con coda lunga a destra

# e dopo la trasformazione la distribuzione si normalizza

# log1p = log plus 1, per evitare la divisione per zero

df.SalePrice = np.log1p(df.SalePrice)



fig, ax = plt.subplots(1,1) # 1,x dove x è il numero di grafici da mostrare

fig.set_figheight(5)        # altezza di tutto il riquadro

fig.set_figwidth(16)        # larghezza di tutto il riquadro

sns.distplot(df['SalePrice'], bins=40, fit=norm)
# Per riconvertire usiamo la funzione inversa che è la trasformazione esponenziale (expm1 = exp minus 1)

df.SalePrice = np.expm1(df.SalePrice)

df.SalePrice.head()
# Normalizziamo tutto

normed_house_train = norm(dummy_house_train)

normed_house_pred = norm(dummy_house_pred)



normed_house_train.head()
print(house_train['SalePrice'].head())

print(normed_house_train['SalePrice'].head())



print("\nRiconvertiamo SalePrice: ")

salePrice = normed_house_train['SalePrice']



salePrice = normReverse(salePrice)

salePrice.head()