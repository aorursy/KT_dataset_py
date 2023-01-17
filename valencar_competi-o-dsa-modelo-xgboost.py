import pandas as pd

import numpy as np

np.set_printoptions(precision=5, suppress=True)

pd.options.display.float_format = '{:.2f}'.format

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/datasetsdsa/dataset_treino.csv')

df.head()
dfteste = pd.read_csv('../input/datasetsdsa/dataset_teste.csv')

PropertyID = dfteste['Property Id'] 

dfteste.head()
len(dfteste.columns)
PropertyID.head()
df.iloc[0, :]
df.describe()
del df['Property Id']
print(df.describe())
len(df)
print(df.describe())
from sklearn import preprocessing

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import shutil

import os



# Encoding dos valores de texto para variáveis nominais

def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = "{}-{}".format(name, x)

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)





# Encoding dos valores de texto para uma única variável dummy. As novas colunas (que não substituem o antigo) terão 1

# em todos os locais onde a coluna original (nome) corresponde a cada um dos valores-alvo. Uma coluna é adicionada para

# cada valor alvo.

def encode_text_single_dummy(df, name, target_values):

    for tv in target_values:

        l = list(df[name].astype(str))

        l = [1 if str(x) == str(tv) else 0 for x in l]

        name2 = "{}-{}".format(name, tv)

        df[name2] = l





# Encoding dos valores de texto para índices (ou seja, [1], [2], [3] para vermelho, verde, azul por exemplo).

def encode_text_index(df, name):

    le = preprocessing.LabelEncoder()

    df[name] = le.fit_transform(df[name].astype(str)) 

    return le.classes_





# Normalização Z-score

def encode_numeric_zscore(df, name, mean=None, sd=None):

    if mean is None:

        mean = df[name].mean()

    if sd is None:

        sd = df[name].std()

    df[name] = (df[name] - mean) / sd





# Converte todos os valores faltantes na coluna especificada para a mediana

import numpy as np

def missing_median(df, name):

    # 'Not Available' -> NaN

    df[name] = df[name].replace('Not Available',np.nan)

    

    med = df[name].median()

    df[name] = df[name].fillna(med)





# Converte todos os valores faltantes na coluna especificada para o padrão

def missing_default(df, name, default_value):

    df[name] = df[name].fillna(default_value)

    



# Converte um dataframe Pandas para as entradas x, y que o TensorFlow precisa

def to_xy(df, target):

    result = []

    for x in df.columns:

        if x != target:

            result.append(x)

    # Descobre o tipo da coluna de destino. 

    target_type = df[target].dtypes

    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type

    # Encoding para int. TensorFlow gosta de 32 bits.

    if target_type in (np.int64, np.int32):

        # Classificação

        dummies = pd.get_dummies(df[target])

        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)

    else:

        # Regressão

        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)



# String de tempo bem formatado

def hms_string(sec_elapsed):

    h = int(sec_elapsed / (60 * 60))

    m = int((sec_elapsed % (60 * 60)) / 60)

    s = sec_elapsed % 60

    return "{}:{:>02}:{:>05.2f}".format(h, m, s)





# Chart de Regressão

def chart_regression(pred,y,sort=True):

    t = pd.DataFrame({'pred' : pred, 'y' : y })  # y.flatten()

    if sort:

        t.sort_values(by=['y'],inplace=True)

    a = plt.plot(t['y'].tolist(),label='expected')

    b = plt.plot(t['pred'].tolist(),label='prediction')

    plt.ylabel('output')

    plt.legend()

    plt.show()



# Remove todas as linhas onde a coluna especificada em +/- desvios padrão

def remove_outliers(df, name, sd):

    drop_rows = df.index[(np.abs(df[name] - df[name].mean()) >= (sd * df[name].std()))]

    df.drop(drop_rows, axis=0, inplace=True)





# Normalização Range

def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1, data_low=None, data_high=None):

    if data_low is None:

        data_low = min(df[name])

        data_high = max(df[name])



    df[name] = ((df[name] - data_low) / (data_high - data_low)) * (normalized_high - normalized_low) + normalized_low
df.iloc[0, :]
nao_relevantes1 = [

'Order',                                   

'Property Name',                                              

'Parent Property Id',                                                             

'Parent Property Name',                                          

'BBL - 10 digits',                                                           

'NYC Borough, Block and Lot (BBL) self-reported',                            

'NYC Building Identification Number (BIN)',                               

'Address 1 (self-reported)',                                             

'Address 2',                                                                           

'Postal Code',                                                                                 

'Street Number',                                                                                

'Street Name',

'Fuel Oil #1 Use (kBtu)',                                                             

'Fuel Oil #2 Use (kBtu)',                                                           

'Fuel Oil #4 Use (kBtu)',                                                             

'Fuel Oil #5 & 6 Use (kBtu)'                                                          

]
categoricos = [

'Borough'                                                                                     

'DOF Gross Floor Area',                                                

'Primary Property Type - Self Selected',                              

'List of All Property Use Types at Property',                                   

'Largest Property Use Type',                                                    

'2nd Largest Property Use Type',                                                    

'2nd Largest Property Use - Gross Floor Area (ft²)',                               

'3rd Largest Property Use Type',                                                     

'3rd Largest Property Use Type - Gross Floor Area (ft²)',                            

'Metered Areas (Energy)',                                                        

'Metered Areas  (Water)'                                                          

]

    

numericos = [ 

'Largest Property Use Type - Gross Floor Area (ft²)',  

'2nd Largest Property Use - Gross Floor Area (ft²)',

'3rd Largest Property Use Type - Gross Floor Area (ft²)', 

'Year Built',                                                                                  

'Number of Buildings - Self-reported',                                                          

'Occupancy', 

'ENERGY STAR Score',                                                                         

'Site EUI (kBtu/ft²)',                                                                      

'Weather Normalized Site EUI (kBtu/ft²)',                                                    

'Weather Normalized Site Electricity Intensity (kWh/ft²)',                                      

'Weather Normalized Site Natural Gas Intensity (therms/ft²)',                                   

'Weather Normalized Source EUI (kBtu/ft²)'                                                  

]
def analisar_distribuicao_variavel(df, var):

    c = df[var].value_counts()

    c = c.to_frame().reset_index()

    return c[:3]
lista = ['Fuel Oil #1 Use (kBtu)',                                                             

'Fuel Oil #2 Use (kBtu)',                                                           

'Fuel Oil #4 Use (kBtu)',                                                             

'Fuel Oil #5 & 6 Use (kBtu)'                                                          

]

for var in lista:

    d = analisar_distribuicao_variavel(df, var)

    print(var)

    print(d)

    print('============================================')
lista = df.columns

lista
# distribuição dos dados - contagem



lista = df.columns

for var in lista:

    d = analisar_distribuicao_variavel(df, var)

    print(var)

    print(d)

    print('============================================')
#* Remover - Atributos não-Relevantes

removeAtributeList = [

'Order',

'Property Name',

'Parent Property Id',

'Parent Property Name',

'BBL - 10 digits',

'NYC Borough, Block and Lot (BBL) self-reported',

'NYC Building Identification Number (BIN)',

'Address 1 (self-reported)',

'Address 2',

'Postal Code',

'Street Number',

'Street Name',

'2nd Largest Property Use - Gross Floor Area (ft²)',

'3rd Largest Property Use Type',

'3rd Largest Property Use Type - Gross Floor Area (ft²)',

'Number of Buildings - Self-reported',

'Occupancy',

'Metered Areas (Energy)',

'Metered Areas  (Water)',

'Fuel Oil #1 Use (kBtu)',

'Fuel Oil #2 Use (kBtu)',

'Fuel Oil #4 Use (kBtu)',

'Fuel Oil #5 & 6 Use (kBtu)',

'Diesel #2 Use (kBtu)',

'District Steam Use (kBtu)',

'Natural Gas Use (kBtu)',

'Weather Normalized Site Natural Gas Use (therms)',

'Release Date',

'DOF Benchmarking Submission Status',

'Latitude',

'Longitude'

]





#* Categorical - tratar- Yes, No, Not Avaliable, NaN

categoricalAtributesList = [

'Borough',

'Primary Property Type - Self Selected',

'List of All Property Use Types at Property',

'Largest Property Use Type',

'2nd Largest Property Use Type',

'Water Required?',

'NTA'

]



#* Numerical - tratar- Yes, No, Not Avaliable, NaN

numericalAtributesList = [

'DOF Gross Floor Area',

'Largest Property Use Type - Gross Floor Area (ft²)',

'Year Built',

'ENERGY STAR Score',

'Site EUI (kBtu/ft²)',

'Weather Normalized Site EUI (kBtu/ft²)',

'Weather Normalized Site Electricity Intensity (kWh/ft²)',

'Weather Normalized Site Natural Gas Intensity (therms/ft²)',

'Weather Normalized Source EUI (kBtu/ft²)',

'Electricity Use - Grid Purchase (kBtu)',

'Weather Normalized Site Electricity (kWh)',

'Total GHG Emissions (Metric Tons CO2e)',

'Direct GHG Emissions (Metric Tons CO2e)',

'Indirect GHG Emissions (Metric Tons CO2e)',

'Property GFA - Self-Reported (ft²)',

'Water Use (All Water Sources) (kgal)',

'Water Intensity (All Water Sources) (gal/ft²)',

'Source EUI (kBtu/ft²)',

'Community Board',

'Council District',

'Census Tract'

]
df.loc[1, 'NTA']

df['NTA'].value_counts()[:2]
df = df[categoricalAtributesList + numericalAtributesList ].copy()

df.head()
numericalAtributesList_test = numericalAtributesList.copy()

numericalAtributesList_test.remove('ENERGY STAR Score')

dfteste = dfteste[categoricalAtributesList + numericalAtributesList_test  ].copy()  

dfteste.head()
len(df.columns)
df.columns
categoricalAtributesList
NULO = -1

for name in categoricalAtributesList:

    df[name] = df[name].fillna(NULO)

    encode_text_index(df, name)

    

    dfteste[name] = dfteste[name].fillna(NULO)

    encode_text_index(dfteste, name)
df.loc[:3, categoricalAtributesList]
df[categoricalAtributesList].describe()
for name in numericalAtributesList:

    missing_median(df, name)

    if name != 'ENERGY STAR Score':

        missing_median(dfteste, name)



df[numericalAtributesList].describe()
dfteste.describe()
classe_x = 'ENERGY STAR Score'

y = df[classe_x]



x_columns = list(df.columns.values)

x_columns.remove(classe_x)



X = df.loc[:, x_columns]

print(len(X), len(x_columns), len(y))
X[:3]
# treino

from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(X)

dftreino = pd.DataFrame(x_scaled, columns=x_columns)

dftreino.head()
# teste

from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(dfteste.values)

dfteste = pd.DataFrame(x_scaled, columns=dfteste.columns)

dfteste.head()
len(dftreino.columns)
X_treino = dftreino.values

y_treino = y.values



X_teste_kaggle = dfteste.values



X_teste_kaggle[:3]

#y[:3]
X_teste_kaggle.shape
# Feature Extraction with RFE

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

# load data

X = X_treino

y = y_treino

# feature extraction

model = LogisticRegression()

rfe = RFE(model, 20)

fit = rfe.fit(X, y)

print("Num Features: ", fit.n_features_)

print("Selected Features: ", fit.support_)

print("Feature Ranking: ", fit.ranking_)

print(dftreino.columns)
dfimp = pd.DataFrame(fit.ranking_, columns=['importancia'])

dfimp['atributo'] = dftreino.columns

dfimp2 = dfimp.sort_values('importancia', ascending=True)

dfimp2.ix[:, ['atributo', 'importancia']]
dfimp2 = dfimp2.reset_index(drop=True)

dfimp2.ix[:19, ['atributo', 'importancia']]
atributos_relevantes = dfimp2.head(20)['atributo'].values

atributos_relevantes
dftreino = dftreino.loc[:, atributos_relevantes]

print(len(dftreino.columns))

dftreino.head()
dfteste = dfteste.loc[:, atributos_relevantes]

print(len(dfteste.columns))

dfteste.head()
X_treino = dftreino.values

y_treino = y



X_teste_kaggle = dfteste.values



X_teste_kaggle[:3]
def gerar_arquivo_kaggle(modelo):

# Gera arquivo para o Kaggle - PropertyId,Score

    nome_arquivo = 'Submissao-v5.1-XGBoost.csv'

    df_saida = pd.DataFrame()

    df_saida['Property Id'] = PropertyID.values

    yteste_previsto = modelo.predict(X_teste_kaggle) 

    yteste_previsto = np.rint(yteste_previsto).astype(np.int64)

    df_saida['score'] =   yteste_previsto.ravel()

    # Salvando o arquivo

    df_saida.to_csv(nome_arquivo, index=False)

    print('Arquivo %s salvo...', nome_arquivo)

    !head Submissao-v5.1-XGBoost.csv
parameters_for_testing = {

   'colsample_bytree':[0.4,0.6,0.8],

    'gamma':[0,0.03,0.1,0.3],

    'min_child_weight':[1.5,6,10],

    'learning_rate':[0.1,0.07],

    'max_depth':[3,5],

    'n_estimators':[300], # 10000

    'reg_alpha':[1e-5, 1e-2,  0.75],

    'reg_lambda':[1e-5, 1e-2, 0.45],

    'subsample':[0.6,0.95]  

}
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

#xgb_model = XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,

#     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, 

#                         nthread=6, scale_pos_weight=1, seed=27)

#gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,

#                        iid=False, verbose=10,scoring='neg_mean_absolute_error') #'neg_mean_absolute_error')

#gsearch1.fit(X_treino, y_treino, eval_metric='mae' )

#print('best params')

#print (gsearch1.best_params_)

#print('best score')

#print (gsearch1.best_score_)
#best params

params = {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.07, 

          'max_depth': 3, 'min_child_weight': 10, 'n_estimators': 300, 

          'reg_alpha': 1e-05, 'reg_lambda': 0.45, 'subsample': 0.6}

#best score

#-9.958036954258406





modelXGB = XGBRegressor(

    colsample_bytree = params['colsample_bytree'], 

    gamma = params['gamma'], 

    learning_rate = params['learning_rate'], 

    max_depths = params['max_depth'],

    min_child_weight = params['min_child_weight'],

    n_estimators = params['n_estimators'],

    reg_alpha =  params['reg_alpha'], 

    reg_lambda = params['reg_lambda'],

    subsample = params['subsample']

)

modelXGB.fit(X_treino, y_treino)



#modelXGB = gsearch1

gerar_arquivo_kaggle(modelXGB)
# verificando o arquivo de saída

dfsaida = pd.read_csv('Submissao-v5.1-XGBoost.csv')

dfsaida.head()
len(dfsaida)
#x = pd.Series(x, name="x variable")

import seaborn as sns

ax = sns.distplot(dfsaida['score'])
dfsaida.score.hist()
dfsaida.score.value_counts()
dfsaida['score'] = dfsaida['score'].apply(lambda w: 1 if w <= 0 else w)

import seaborn as sns

ax = sns.distplot(dfsaida['score'])
dfsaida.score.value_counts()
nome_arquivo = 'Submissao-v5.1-XGBoost.csv'

dfsaida.to_csv(nome_arquivo, index=False)

print('Arquivo %s salvo...', nome_arquivo)

!head Submissao-v5.1-XGBoost.csv
ax = sns.distplot(dfsaida['score'])