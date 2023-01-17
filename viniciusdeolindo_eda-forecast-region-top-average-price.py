import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/avocado-prices/avocado.csv")
df['Date'] = pd.to_datetime(df['Date'])
## Estatística descritiva
round(df[['AveragePrice','Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags']].describe(),2)
# Relatório dos dados via pandas profiling
profile = df.profile_report(title='Report - Avocados')
profile
# Agregação de variaveis por soma
a = df.drop(columns=['AveragePrice'])
b = a.groupby(['Date','region','year','type']).sum().reset_index()

# Agregação de variaveis por média
c = df[['Date','region','year','type','AveragePrice']]
d = c.groupby(['Date','region','year','type']).mean().round(2).reset_index()

# Junção das agregações
df = pd.merge(b,d,on=['Date','region','year','type'])
# Top 5 preços médios mais caros por região
region = df.filter(['region','AveragePrice'])
a = region.groupby(['region']).mean().round(2).sort_values(by = 'AveragePrice',ascending=False).reset_index().head(5)
a
# Top 5 preços médios mais caros por região (Boxplot)
b1 = list(a['region'].unique())
box = df[(df['region'] == b1[0]) | (df['region'] == b1[1]) | (df['region'] == b1[2]) | (df['region'] == b1[3]) | (df['region'] == b1[4])]
plt.figure(figsize=(10,8))
sns.boxplot(x = box["region"],y = box["AveragePrice"], palette="Blues")
plt.show()
# Top 5 preços médios mais baratos por região
region = df.filter(['region','AveragePrice'])
c = region.groupby(['region']).mean().round(2).sort_values(by = 'AveragePrice',ascending=True).reset_index().head(5)
c
# Top 5 preços médios mais baratos por região (Boxplot)
d1 = list(c['region'].unique())
box2 = df[(df['region'] == d1[0]) | (df['region'] == d1[1]) | (df['region'] == d1[2]) | (df['region'] == d1[3]) | (df['region'] == d1[4])]
plt.figure(figsize=(10,8))
sns.boxplot(x = box2["region"],y = box2["AveragePrice"], palette="Greens")
plt.show()
# Regiões top 5 preços médios mais caros:
regions = b1

# Construção de gráficos:
for index in range(len(regions)):
    # Transformações de dados e agregação das informações:
    a = df.loc[df['region'] == regions[index]]
    b = a[['Date','AveragePrice']]
    c = b.groupby('Date')['AveragePrice'].mean().reset_index()
        
    # Criação dos gráficos:
    fig, ax = plt.subplots(figsize=(18, 5))
    
    # Construção dos gráficos de linhas:
    ax.plot(c['Date'], c['AveragePrice'])
    
    # Titulos e rótulos
    ax.set_title("Preços médios dos abacates por região: %s" % (regions[index]))
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço médio dos abacates')
# Regiões top 5 preços médios mais baratos:
regions = d1

# Construção de gráficos:
for index in range(len(regions)):
    # Transformações de dados e agregação das informações:
    a = df.loc[df['region'] == regions[index]]
    b = a[['Date','AveragePrice']]
    c = b.groupby('Date')['AveragePrice'].mean().reset_index()
        
    # Criação dos gráficos:
    fig, ax = plt.subplots(figsize=(18, 5))
    
    # Construção dos gráficos de linhas:
    ax.plot(c['Date'], c['AveragePrice'])
    
    # Titulos e rótulos
    ax.set_title("Preços médios dos abacates por região: %s" % (regions[index]))
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço médio dos abacates')
# Dataset da região top região com o preço mais caro:
df = df[df['region'] == b1[0]]
# Tratamento do nome das colunas:
df.rename(columns={'Total Volume':'Total_Volume','Total Bags':'Total_Bags',
                   'Small Bags':'Small_Bags','Large Bags': 'Large_Bags',
                   'XLarge Bags': 'XLarge_Bags'},inplace=True)

# Recursos com a variável de data:
df['day'] = df['Date'].dt.day
df['weekday'] = df['Date'].dt.weekday
df['month'] = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter

# Estatísticas móveis do volume total:
df['mm7_tvol'] = df['Total_Volume'].rolling(7).mean()
df['mm14_tvol'] = df['Total_Volume'].rolling(14).mean()
df['sm7_tvol'] = df['Total_Volume'].rolling(7).std()
df['sm14_tvol'] = df['Total_Volume'].rolling(14).std()

# Estatísticas móveis do preço médio:
df['mm7_avp'] = df['AveragePrice'].rolling(7).mean()
df['mm14_avp'] = df['AveragePrice'].rolling(14).mean()
df['sm7_avp'] = df['AveragePrice'].rolling(7).std()
df['sm14_avp'] = df['AveragePrice'].rolling(14).std()

# Recursos com o volume total
df['plu_4046%'] = round(df['4046']/df['Total_Volume'],3)
df['plu_4225%'] = round(df['4225']/df['Total_Volume'],3)
df['plu_4770%'] = round(df['4770']/df['Total_Volume'],3)

# Recursos com a total de sacolas
df['small_bags%'] = round(df['Small_Bags']/df['Total_Bags'],3)
df['large_bags%'] = round(df['Large_Bags']/df['Total_Bags'],3)
df['xlarge_bags%'] = round(df['XLarge_Bags']/df['Total_Bags'],3)

# Recursos com volume total do abacate
df['vol_bin'] = pd.qcut(df['Total_Volume'], 4, labels=False)
df['vol_bin'] = df['vol_bin'].round(0).astype(str)

# Recursos com volume de sacolas do abacate
df['bags_bin'] = pd.qcut(df['Total_Bags'], 4, labels=False)
df['bags_bin'] = df['bags_bin'].round(0).astype(str)

# Recursos com o tipo de abacate
df = pd.get_dummies(df, columns=['type','vol_bin','bags_bin','quarter'])

# Tratamento de nulos
df = df.fillna(value = 0)
# Pacotes de aprendizagem de máquina do pacote sklearn: Gradient Boosting Regressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
## Mapa de correlações
# Construção do mapa de correlações
corr = round(df.corr(method = 'spearman'),2)
plt.figure(figsize=(10,8))
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
plt.title("Correlações entre variáveis pelo método spearman")
plt.show()
## Seleção de variáveis via correlação de Spearman
# Calculo de correlação de Spearman com o foco na variável target
cor_target = abs(corr['AveragePrice'])

# Seleção de variáveis na base
relevant_features = cor_target[cor_target>0.5]
corr2 = pd.DataFrame(relevant_features).T
sel_corr_sper = list(corr2.columns)
sel_corr_sper.append("Date")
sel_corr_sper.append("region")

# Base com variáveis selecionadas
df = df.filter(sel_corr_sper)
pd.DataFrame({'Variáveis para modelagem':list(df)})
## Modelagem do preço médio na região mais cara: HartfordSpringfield
# Divisão do dataset:
treino1 = df[df['Date'] < "2018-01-20"]
teste1 = df[df['Date'] >= "2018-01-20"]
 
# Tratamento dos dados de treino e de teste:
Xtr, ytr = treino1.drop(['Date','region','AveragePrice'],axis=1),treino1['AveragePrice']
Xval, yval = teste1.drop(['Date','region','AveragePrice'],axis=1),teste1['AveragePrice']

# Padronização dos dados de treino:
min_max=MinMaxScaler()
Xtrm=min_max.fit_transform(Xtr)
Xvalm=min_max.fit_transform(Xval)

# Grid search with cross validation:
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits=10, random_state=8)
model = GradientBoostingRegressor(random_state = 7)

# Grid search: parâmetros
param_grid = {
   "n_estimators": [20,50,100],
   'learning_rate': [0.04, 0.03, 0.01],
   'max_depth': [3,4,5],
   'min_samples_split': [0.0050, 0.0040, 0.0035, 0.0010],
   'subsample':[0.6,0.7,0.8,0.9],
   'max_features': ['sqrt', 'log2']
   }
# Execução do grid search
CV_model = GridSearchCV(estimator=model,param_grid=param_grid,cv=kfold,scoring=scoring)
CV_model_result = CV_model.fit(Xtrm, ytr)

# Print resultados
print("Resultados do treinamento do modelo via pesquisa cartesiana:")
print(" ")
print("Melhor resultado da pesquisa: %f usando os parâmetros %s" % (CV_model_result.best_score_, CV_model_result.best_params_))

# Treino do modelo
baseline = GradientBoostingRegressor(**CV_model_result.best_params_)
baseline.fit(Xtrm,ytr)

# Previsão
p = baseline.predict(Xvalm)

# Avaliação da previsão
a = pd.Series(p)
b = pd.Series(yval)
x = {'Previsto': a} 
y = {'Realizado': b} 
w = pd.DataFrame(x)
z = pd.DataFrame(y)

# Dados para gráfico da previsão
df_teste = pd.concat([w.reset_index(drop=True), z.reset_index()], axis=1)
graf = df_teste[["Previsto","Realizado"]].round(2)

# Metricas
print("Métricas para avaliação do desempenho do modelo GBM: ")
print(" ")

# MAE
errors = abs(graf['Previsto'] - graf['Realizado'])
print('MAE: ',round(np.mean(errors), 2))

# MSE
meanSquaredError=mean_squared_error(yval, p)
print("MSE:", round(meanSquaredError,2))

# RMSE
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", round(rootMeanSquaredError,2))

# MAPE
mape = 100 * (errors/graf['Realizado'])
print("MAPE:", round(np.mean(mape),2), '%.')

# Accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Gráfico
print("Gráfico de desempenho da previsão:")
print(" ")
graf.plot(figsize=(18, 5),title='Gráfico de linhas - Previsto e Realizado',grid=True)
plt.show()

# Tabela
print("Tabela de desempenho da previsão:")
print(" ")
df_teste[["Previsto","Realizado"]].round(2).T