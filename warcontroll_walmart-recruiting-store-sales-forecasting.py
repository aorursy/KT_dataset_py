import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from time import time


#IMPORTAÇÃO DO DATASET FEATURES
df_features = pd.read_csv("features.csv",sep=',')
df_features.head(5)
#VERIFICAÇÃO DE CAMPOS NULOS
print(df_features.count())
#IMPORTAÇÃO DATASET STORE
df_stores = pd.read_csv("stores.csv",sep=',')
df_stores.head(5)
#VERIFICAÇÃO DE CAMPOS NULOS
print(df_stores.count())
#IMPORTAÇÃO DATASET DE TREINO
df_train = pd.read_csv("train.csv",sep=',')
df_train.head(5)
#VERIFICAÇÃO DE CAMPOS NULOS
print(df_train.count())
#UNIÃO DATASET DE FEATURES COM DATASET STORES
df_feat_sto = df_features.merge(df_stores, how='inner', on='Store')
#VERIFICAR SE NÃO OCORREU CARTESIANO
df_feat_sto.count()
#UNIÃO DADOS DE FEATURES + STORES + TRAIN
df_train_detail = df_train.merge(df_feat_sto, 
                           how='inner',
                           on=['Store','Date','IsHoliday']).sort_values(by=['Store',
                                                                            'Dept',
                                                                            'Date']).reset_index(drop=True)
#VERIFICAR SE NÃO OCORREU CARTESIANO
df_train_detail.count()
#VERIFICANDO O TIPO DE DADOS DAS COLUNAS
df_train_detail.dtypes
#CRIA A COLUNA MÊS
df_train_detail['month'] = df_train_detail['Date'].apply(lambda x: x[5:7])
df_train_detail['month'] = df_train_detail['month'].astype('int64')
#CRIA A COLUNA ANO
df_train_detail['year'] = df_train_detail['Date'].apply(lambda x: x[0:4])
df_train_detail['year'] = df_train_detail['year'].astype('int64')
#TRANSFORMAR COLUNA DATE EM DATETIME - FORMATO DE DATAS
df_train_detail['Date'] = pd.to_datetime(df_train_detail['Date'])
df_train_detail['weekday'] = df_train_detail['Date'].apply(lambda d: d.weekday())
#QUANTIDADE DE CAMPOS NULOS EM AMARELO
sns.heatmap(df_train_detail.isnull(), yticklabels=False, cbar=False,cmap="viridis")
#CRIA DICIONÁRIO PARA MEIO DE SEMANA E FIM DE SEMANA - PROMOCOES SOMENTE DE SEXTA-FEIRA
# 0 Workday
# 1 weekend
week_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1}
#TRANSFORMAR INDEX EM COLUNA 
df_train_detail['coluna'] = df_train_detail.index
#ORDENAR POR DATA
df_train_detail = df_train_detail.sort_values(['Date'])
#VALORES VENDAS ANUAIS
round(df_train_detail.groupby(['year']).agg({'Weekly_Sales':['sum','mean','count']}),2)
#VALORES VENDAS MENSAIS
round(df_train_detail.groupby(['year','month']).agg({'Weekly_Sales':['sum','mean','count']}),2)
#VALORES VENDAS POR SEMANA
round(df_train_detail.groupby(['Date']).agg({'Weekly_Sales':['sum','mean','count']}),2).reset_index()
#ANÁLISE DA MÉDIA E DA MEDIANA POR SEMANA
weekly_sales_mean = df_train_detail['Weekly_Sales'].groupby(df_train_detail['Date']).mean()
weekly_sales_median = df_train_detail['Weekly_Sales'].groupby(df_train_detail['Date']).median()
plt.figure(figsize=(20,8))
sns.lineplot(weekly_sales_mean.index, weekly_sales_mean.values)
sns.lineplot(weekly_sales_median.index, weekly_sales_median.values)
plt.grid()
plt.legend(['Mean', 'Median'], loc='best', fontsize=16)
plt.title('Weekly Sales - Mean and Median', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.show()
weekly_sales = df_train_detail['Weekly_Sales'].groupby(df_train_detail['Store']).sum()
plt.figure(figsize=(20,8))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Total Sales - per Store', fontsize=18)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Store', fontsize=16)
plt.show()
#VALORES VENDAS POR MÊS E ANO - SOMENTE FERIADOS
round(df_train_detail[df_train_detail['IsHoliday']==True].groupby(['Date']).agg({'Weekly_Sales':['mean']}),2).reset_index()
#VALORES VENDAS POR MES E ANO - SOMENTE NÃO FERIADO
round(df_train_detail[df_train_detail['IsHoliday']==False].groupby(['Date']).agg({'Weekly_Sales':['mean']}),2).reset_index()
#VALORES VENDAS POR MES E ANO - TOTAL
round(df_train_detail.groupby(['Date']).agg({'Weekly_Sales':['mean']}),2).reset_index()
#EXCLUSÃO DE DADOS CATEGORICOS - São dados que não podem ser realizdos calculos, mas para frente verei se poderemos inclui lo de outras formas
df_train_detail = df_train_detail.drop(columns=['CPI']) # indice de preço ao consumidor
df_train_detail = df_train_detail.drop(columns=['Temperature']) # Temperaura
df_train_detail = df_train_detail.drop(columns=['Unemployment']) # Taxa de desemprego
#df_train_detail = df_train_detail.drop(columns=['Coluna']) # Coluna extra crida apenas para ordenação
#ANÁLISE DE CORRELAÇÃO DAS VARIÁVEIS
sns.set(style="white")

corr = df_train_detail.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Definindo a plotagem no matplotlib
f, ax = plt.subplots(figsize=(20, 15))

# Gerando o mapa de cores a partir do seaborn
cmap = sns.diverging_palette(220, 10, as_cmap=True)

plt.title('Correlation Matrix', fontsize=18)
# Desenhando o heatmap com a mascara
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()
corr.head(10)
#AJUSTANDO CAMPOS NULOS
df_train_detail = df_train_detail.fillna(0)
#IMPORT DATASET DE TREINO
df_test = pd.read_csv('test.csv')
#VERIFICAÇÃO DE DATA MÁXIMA
data_max = datetime.strptime('2013-07-26', '%Y-%m-%d')
#VERIFICAÇÃO DE DATA MINIMA
data_mim = datetime.strptime('2012-11-02', '%Y-%m-%d')
#CALCULO DE SEMANAS A PREVER
semanas = (data_max - data_mim)
print(266//7)
#TESTANDO MODELO DE SERIES TEMPORARIS - Random-Forest
df_train_detail.head(5)
# Obtém todas as colunas do dataframe
colunas = df_train_detail.columns.tolist()
# Filtra as colunas e remove as que não são relevantes
colunas = [c for c in colunas if c not in ["coluna", "Fuel_Price",'Date','Type','Weekly_Sales']]
# Preparando a variável target, a que será prevista
target = "Weekly_Sales"
# Gerando os dados de treino
df_treino = df_train_detail.sample(frac = 0.8, random_state = 101)
# Seleciona tudo que não está no dataset de treino e armazena no dataset de teste
df_teste = df_train_detail.loc[~df_train_detail.index.isin(df_treino.index)]
# Shape dos datasets
print(df_treino.shape)
print(df_teste.shape)
# Criando um Regressor
reg_v1 = LinearRegression()
# Fit the model to the training data.
modelo_v1 = reg_v1.fit(df_treino[colunas], df_treino[target])
# Fazendo previsões
previsoes = modelo_v1.predict(df_teste[colunas])
# Computando os erros entre valores observados e valores previstos
mean_squared_error(previsoes, df_teste[target])
# Criando um regressor Random Forest
reg_v2 = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 10, random_state = 1)
# Criando o modelo
modelo_v2 = reg_v2.fit(df_treino[colunas], df_treino[target])
# Fazendo previsões
previsoes = modelo_v2.predict(df_teste[colunas])
# Computando o erro
mean_squared_error(previsoes, df_teste[target])
#IMPORT DATASET DE TEST WALLMART
df_test_wall_mart = pd.read_csv('test.csv')
df_test_wall_mart.head(5)
#UNIÃO DADOS DE FEATURES + STORES + TRAIN
df_test_wall_mart = df_test_wall_mart.merge(df_feat_sto, 
                           how='inner',
                           on=['Store','Date','IsHoliday']).sort_values(by=['Store',
                                                                            'Dept',
                                                                            'Date']).reset_index(drop=True)
#CRIA A COLUNA MÊS
df_test_wall_mart['month'] = df_test_wall_mart['Date'].apply(lambda x: x[5:7])
df_test_wall_mart['month'] = df_test_wall_mart['month'].astype('int64')
#CRIA A COLUNA ANO
df_test_wall_mart['year'] = df_test_wall_mart['Date'].apply(lambda x: x[0:4])
df_test_wall_mart['year'] = df_test_wall_mart['year'].astype('int64')
#TRANSFORMAR COLUNA DATE EM DATETIME - FORMATO DE DATAS
df_test_wall_mart['Date'] = pd.to_datetime(df_test_wall_mart['Date'])
df_test_wall_mart['weekday'] = df_test_wall_mart['Date'].apply(lambda d: d.weekday())
#AJUSTANDO CAMPOS NULOS
df_test_wall_mart = df_test_wall_mart.fillna(0)
# Obtém todas as colunas do dataframe
colunas_2 = df_test_wall_mart.columns.tolist()
# Filtra as colunas e remove as que não são relevantes
colunas_2 = [c for c in colunas if c not in ["Fuel_Price",'Date','Type']]
df_test_wall_mart[['Store','Dept','IsHoliday','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Size','month','year','weekday']]
df_test_wall_mart.head(5)
previsoes_test = modelo_v2.predict(df_test_wall_mart[['Store','Dept','IsHoliday','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Size','month','year','weekday']])
df_test_wall_mart["Weekly_Sales"] =  previsoes_test
#FORMATAR PARA SUBMETER
df_test_wall_mart['id'] = df_test_wall_mart['Store'].astype(str)+'_'+df_test_wall_mart['Dept'].astype(str)+'_'+df_test_wall_mart['Date'].astype(str)
#GERAÇÃO DE DATAFRAME NO FORMATO DE SUBMETER
df_test_wall_mart[['id','Weekly_Sales']]