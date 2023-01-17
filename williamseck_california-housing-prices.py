import pandas as pd

import numpy as np

%pylab inline

pd.set_option("display.max_columns", None)

pd.set_option("display.max_rows", None)

import warnings; warnings.simplefilter('ignore')
df_raw = pd.read_csv('../input/housing.csv')
df_raw.head()
# Recherches des valeurs nulles

df_raw.info()       # les données sont complètes. pas de valeurs manquantes
#données manquantes ou valeurs nulles

pd.isnull(df_raw).sum()
df_raw['ocean_proximity'].value_counts()
# taille du data frame

df_raw.shape
ocean_proximity_dummies_houses  = pd.get_dummies(df_raw['ocean_proximity'])

ocean_proximity_dummies_houses.columns = ['<1H OCEAN','INLAND','NEAR OCEAN','NEAR BAY','ISLAND']
ocean_proximity_dummies_houses.head()
# On élimine la variable 'ocean_proximity'

df_raw.drop(['ocean_proximity'],axis=1,inplace=True)
# On remplace la variable 'ocean_proximity' par ses sous-variables obtenues par dummification

df_raw = df_raw.join(ocean_proximity_dummies_houses)
# On visualise le nouveau tableau 

df_raw.tail()
# package permettant de normaliser les variables (voir à la fin)

from sklearn.preprocessing import MinMaxScaler
df_raw['income_cat']=np.ceil(df_raw['median_income'] /1.5)

df_raw['income_cat'].where(df_raw['income_cat'] < 5, 5.0 , inplace = True)
# diagramme en battons pour le revenu median

df_raw['income_cat'].value_counts().plot(kind='bar')

plt.title('revenu median par catégorie')

plt.xlabel('catégorie')

plt.ylabel('montant')
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state = 42)
for train_index, test_index in split.split(df_raw,df_raw['income_cat']):

    strat_train_set = df_raw.loc[train_index]  # L'indexeur de locus Pandas peut être utilisé avec DataFrames pour deux 

                                          # cas d'utilisation différents:

                                          # a.) Sélection des lignes par étiquette / index

                                          # b.) Sélection de lignes avec une recherche booléenne / conditionnelle

    strat_test_set =df_raw.loc[test_index]
housing = strat_train_set.copy()
# On élimine la variable 'median_income'

housing.drop(['income_cat'],axis=1,inplace=True)
housing.head()
df_raw.plot(kind='scatter', x='longitude',y='latitude', alpha = 0.1)

# on ajoute alpha= 0,1 pour distinguer les points de plus forte densité de population
df_raw.plot(kind='scatter', x='longitude',y='latitude', alpha = 0.4,

             s=df_raw['population']/100, label='population', figsize=(10,7),

             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar = True,)

plt.legend()
import seaborn as sns

sns.set_style('whitegrid')
# Matrice couleur des données

def plot_correlation_map( df_raw ):

    corr = df_raw.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )

plot_correlation_map(df_raw)
corr_matrix = df_raw.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

# Corrélation positive forte => les prix augmentent. négative => les prix diminuent. proche de zéro = pas de corr

# UNIQUEMENT POUR LES CORRELATIONS LINEAIRES
# Nombre de chambres par appartement:

df_raw['room_per_households']=df_raw['total_rooms']/df_raw['households']

df_raw['bedrooms_per_room']=df_raw['total_bedrooms']/df_raw['total_rooms']

df_raw['population_per_house']=df_raw['population']/df_raw['households']
corr_matrix = df_raw.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)
df_raw.head()
#données manquantes 

pd.isnull(df_raw).sum()
# On remplace les valeurs manquantes par la médiane

df_raw["total_bedrooms"].fillna(df_raw["total_bedrooms"].median(), inplace=True)

# convert from float to int

df_raw['total_bedrooms'] = df_raw['total_bedrooms'].astype(int)
# On remplace les valeurs manquantes par la médiane

df_raw["bedrooms_per_room"].fillna(df_raw["bedrooms_per_room"].median(), inplace=True)

# convert from float to int

df_raw['bedrooms_per_room'] = df_raw['bedrooms_per_room'].astype(int)
# normalisation de certaines variables

df_raw['longitude'] = MinMaxScaler().fit_transform(df_raw['longitude'].values.reshape(-1, 1))

df_raw['latitude'] = MinMaxScaler().fit_transform(df_raw['latitude'].values.reshape(-1, 1))   

df_raw['total_rooms'] = MinMaxScaler().fit_transform(df_raw['total_rooms'].values.reshape(-1, 1))   

df_raw['population'] = MinMaxScaler().fit_transform(df_raw['population'].values.reshape(-1, 1))   

df_raw['households'] = MinMaxScaler().fit_transform(df_raw['households'].values.reshape(-1, 1))   

df_raw['total_bedrooms'] = MinMaxScaler().fit_transform(df_raw['total_bedrooms'].values.reshape(-1, 1)) 

df_raw['median_house_value'] = (df_raw['median_house_value']/100).astype(int)
df_raw.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# définition de la variable 'cols' (plus facile pour manipuler toutes les colonnes)

cols = ['longitude', 

        'latitude',

        'housing_median_age',

        'total_rooms',

        'total_bedrooms',

        'population',

        'households',

        'median_income',

        'median_house_value',

        '<1H OCEAN',

        'INLAND',

        'NEAR OCEAN',

        'NEAR BAY',

        'ISLAND',

        'room_per_households',

        'bedrooms_per_room',

        'population_per_house'

       ] 
# X représente les colonnes

X = df_raw[cols]
# y représente la colonne 'median_house_value'. 

# On élimine 'median_house_value' de X

y = X['median_house_value']

del X['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVR              # Pour des valeurs continues en cible on va utiliser SVR

svr = SVR()

svr.fit(X_train, y_train)

print(svr.score(X_train, y_train))

print(svr.score(X_test, y_test))
def parse_model_0(X):

    target = X.median_house_value

    X=X[cols] #(Valeurs ayant des données complètes)😊

    return X, target
X,y = parse_model_0(df_raw.copy())
from sklearn.model_selection import cross_val_score

def compute_score(clf, X, y):

    xval = cross_val_score(clf, X, y, cv = 5)

    return mean(xval) 