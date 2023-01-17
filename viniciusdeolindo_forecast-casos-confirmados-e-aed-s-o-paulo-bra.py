# Pacotes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_profiling
import seaborn as sns
from datetime import date
# Entrada de dados
df = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')
df['date'] = pd.to_datetime(df['date'])
# Dados do estado de São Paulo
sp = df[df['state'] == 'São Paulo']
print("Informações básicas do dataset")
print("Última data de atualização: ",df['date'].max())
print("Número de casos confirmados: ",df['cases'].max())
print("Número de obitos: ",df['deaths'].max())
sp.describe().round(2)
# Entrada de dados
sp2 = pd.read_csv('../input/sp02062020/covid19-282b9c496d354728a965294c90962764.csv')
# Analise exploratoria de dados
sp2[['deaths_covid19','new_deaths_covid19']].describe().round(2)
# Transformação das informações
sp2['date'] = pd.to_datetime(sp2['date'])
sp2 = sp2.filter(['date','state','epidemiological_week_2020','deaths_covid19','new_deaths_covid19'])
sp2 = sp2.sort_values(by=['date'])
sp2.rename(columns={'state': 'uf'}, inplace=True)
sp2 = sp2[(sp2['date'] <= date.today().strftime("%Y-%m-%d"))]
# Informações básicas do dataset do registro civil
a = sp2.loc[sp2['new_deaths_covid19'] == sp2['new_deaths_covid19'].max()].filter(['date'])

print("Informações básicas do dataset do registro civil")
print("Última data de atualização: ",sp2['date'].max())
print("Número de obitos registrados: ",sp2['new_deaths_covid19'].sum())
print("Número maximo de registros de obitos: ", sp2['new_deaths_covid19'].max())
print("Data do número maximo de registros de obitos: ", a['date'].max())
print("Média de obitos registrados: ",round(sp2['new_deaths_covid19'].mean(),0))
# Informações de data
sp['month'] = sp['date'].dt.month
sp['weekday'] = sp['date'].dt.weekday
sp['day'] = sp['date'].dt.day

# Estatística de casos
sp['cases_quant'] = sp['cases'].diff()
sp['cases_mm3'] = sp['cases_quant'].rolling(3).mean()
sp['cases_mm7'] = sp['cases_quant'].rolling(7).mean()
sp['cases_sd3'] = sp['cases_quant'].rolling(3).std()
sp['cases_sd7'] = sp['cases_quant'].rolling(7).std()

# Estatísticas de número de mortes
sp['deaths_quant'] = sp['deaths'].diff()
sp['deaths_mm3'] = sp['deaths_quant'].rolling(3).mean()
sp['deaths_mm7'] = sp['deaths_quant'].rolling(7).mean()
sp['deaths_sd3'] = sp['deaths_quant'].rolling(3).std()
sp['deaths_sd7'] = sp['deaths_quant'].rolling(7).std()

# Taxa de letalidade
sp['tx_casos'] = round(sp['deaths_quant']/sp['cases_quant'],2)
sp['tx_casos_mm3'] = sp['tx_casos'].rolling(3).mean()
sp['tx_casos_mm7'] = sp['tx_casos'].rolling(7).mean()

# Sinalização das informações das restrições do estado de São Paulo
m = (sp['date']) >(pd.to_datetime('2020-03-24') - datetime.timedelta(1*365/12))
sp['flag_quarentena'] = m.astype(int)

n = (sp['date']) >(pd.to_datetime('2020-05-11') - datetime.timedelta(1*365/12))
sp['flag_rodizio'] = n.astype(int)

p = (sp['date']) >(pd.to_datetime('2020-05-07') - datetime.timedelta(1*365/12))
sp['flag_mascara'] = p.astype(int)

q = (sp['date']) >(pd.to_datetime('2020-04-27') - datetime.timedelta(1*365/12))
sp['flag_aux_emg'] = p.astype(int)

# Estatísticas dos registros de obitos no registro civil
sp2['new_death_mm3'] = sp2['new_deaths_covid19'].rolling(3).mean()
sp2['new_death_mm7'] = sp2['new_deaths_covid19'].rolling(7).mean()
sp2['new_death_sd3'] = sp2['new_deaths_covid19'].rolling(3).std()
sp2['new_death_sd7'] = sp2['new_deaths_covid19'].rolling(7).std()
# Enriquecimento da base com a base de registro civil
base = pd.merge(sp,sp2, on = 'date')
base = base.filter(['date','epidemiological_week_2020','month','weekday','day','region','state','uf','cases','cases_quant','cases_mm3',
 'cases_mm7','cases_sd3','cases_sd7','deaths','deaths_quant','deaths_mm3','deaths_mm7','deaths_sd3','deaths_sd7','tx_casos','tx_casos_mm3',
 'tx_casos_mm7','flag_quarentena','flag_rodizio','flag_mascara','flag_aux_emg','deaths_covid19','new_deaths_covid19','new_death_mm3',
 'new_death_mm7','new_death_sd3','new_death_sd7'])
# Correlações
corr = round((base).corr(method = 'spearman'),2)
plt.figure(figsize=(8,6))
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
plt.title("Correlações entre os casos de e ações de governo covid-19 no estado de São Paulo")
plt.show()
# Tratamento de nulos
base = base.fillna(value = 0)
# Gráfico acumulativo
plt.figure(figsize=(12,6))
plt.plot(base['date'],base['cases'])
plt.plot(base['date'],base['deaths'])
plt.title("Gráfico de crescimento do covid-19 no estado de São Paulo")
plt.legend(['Casos', 'Mortes'])
plt.axvline(pd.to_datetime('2020-03-24'), color='r', linestyle='--', lw=2) #ínicio da quarentena
plt.axvline(pd.to_datetime('2020-04-27'), color='g', linestyle='--', lw=2) #inicio da obrigação do uso de mascaras
plt.axvline(pd.to_datetime('2020-05-07'), color='y', linestyle='--', lw=2) #inicio do pagamento do auxilio emergencial
plt.show()
# Gráfico quantitativo
plt.figure(figsize=(12,6))
plt.plot(base['date'],base['cases_quant'])
plt.plot(base['date'],base['cases_mm3'])
plt.plot(base['date'],base['cases_mm7'])
plt.plot(base['date'],base['deaths_quant'])
plt.title("Gráfico de crescimento do covid-19 no estado de São Paulo")
plt.axvline(pd.to_datetime('2020-03-24'), color='r', linestyle='--', lw=2) #ínicio da quarentena
plt.axvline(pd.to_datetime('2020-04-27'), color='g', linestyle='--', lw=2) #inicio da obrigação do uso de mascaras
plt.axvline(pd.to_datetime('2020-05-07'), color='y', linestyle='--', lw=2) #inicio do pagamento do auxilio emergencial
plt.legend(['Casos', 'Casos MM3','Casos MM7','Mortes'])
plt.show()
# Gráfico quantitativo
plt.figure(figsize=(12,6))
plt.plot(base['date'],base['tx_casos'])
plt.plot(base['date'],base['tx_casos_mm3'])
plt.plot(base['date'],base['tx_casos_mm7'])
plt.title("Gráfico da taxa de letalidade do covid-19 no estado de São Paulo")
plt.axvline(pd.to_datetime('2020-03-24'), color='r', linestyle='--', lw=2) #ínicio da quarentena
plt.axvline(pd.to_datetime('2020-04-27'), color='g', linestyle='--', lw=2) #inicio da obrigação do uso de mascaras
plt.axvline(pd.to_datetime('2020-05-07'), color='y', linestyle='--', lw=2) #inicio do pagamento do auxilio emergencial
plt.legend(['Taxa de letalidade'])
plt.show()
# Gráfico de obitos por registro civil
plt.figure(figsize=(12,6))
plt.bar(base['date'],base['new_deaths_covid19'])
plt.plot(base['date'],base['new_death_mm3'])
plt.plot(base['date'],base['new_death_mm7'])
plt.title("Gráfico de registro de obitos no dia de covid-19 no estado de São Paulo")
plt.axvline(pd.to_datetime('2020-03-24'), color='r', linestyle='--', lw=2) #ínicio da quarentena
plt.axvline(pd.to_datetime('2020-04-27'), color='g', linestyle='--', lw=2) #inicio da obrigação do uso de mascaras
plt.axvline(pd.to_datetime('2020-05-07'), color='y', linestyle='--', lw=2) #inicio do pagamento do auxilio emergencial
#plt.legend(['Nº de registro de obitos - via data de obito'])
plt.show()
# Gráfico de obitos por registro civil por semana epidemi
plt.figure(figsize=(12,6))
plt.bar(base['epidemiological_week_2020'],base['new_deaths_covid19'])
plt.title("Gráfico de registro de obitos por semana epidemiologicas de covid-19 no estado de São Paulo")
plt.legend(['Nº de registro de obitos - via data de obito'])
plt.show()
# Modelagem supervisionada - Regressão
from sklearn.feature_selection import mutual_info_regression, SelectKBest,f_regression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
# Tratamento de dados
base['weekday'] = pd.Series(base['weekday'], dtype="string")
week = pd.get_dummies(base["weekday"])
del base['weekday']
week.rename(columns={0:'domingo',1:'segunda',2:'terca',3:'quarta',4:'quinta',5:'sexta',6:'sabado'},inplace=True)
base = pd.concat([base,week],axis=1)
base.shape
# Filtro de dados 
base = base.filter(['cases_quant','cases_mm3','cases_mm7','cases_sd3', 'cases_sd7','deaths_mm3','deaths_mm7',
 'deaths_sd3','deaths_sd7','flag_rodizio', 'flag_mascara','flag_aux_emg','date','domingo', 'segunda', 'terca',
 'quarta', 'quinta', 'sexta', 'sabado'])
# Seleção de recursos via correlação de spearman
cor = base.corr(method = 'spearman')
cor_target = abs(cor['cases_quant'])
relevant_features = cor_target[cor_target>0.5]
corr2 = pd.DataFrame(relevant_features).T
sel_corr_sper = list(corr2.columns)
sel_corr_sper.append("date")
# Seleção de recursos via correlação de spearman
base = base.filter(sel_corr_sper)

# Retirando as datas sem casos
base = base[base['date']>"2020-03-03"]
# Divisão da base de dados
treino1 = base[base['date'] < "2020-05-29"]
teste1 = base[base['date'] >= "2020-05-29"]
treino1.shape, teste1.shape
# Divisão entre os regressores e a variavel target
Xtr, ytr = treino1.drop(['cases_quant','date'], axis=1), treino1['cases_quant']
Xval, yval = teste1.drop(['cases_quant','date'], axis=1), teste1['cases_quant']
Xtr.shape, Xval.shape
# Pré-processo MinMax
min_max=MinMaxScaler()
Xtrm=min_max.fit_transform(Xtr)
Xvalm=min_max.fit_transform(Xval)
# Modelos para seleção
models = [] 
models.append(('ADA', AdaBoostRegressor())) 
models.append(('GBM', GradientBoostingRegressor())) 
models.append(('RFT', RandomForestRegressor())) 
models.append(('CART', DecisionTreeRegressor())) 

# Avalia os algoritmos
results = [] 
names = [] 

for name, model in models: 
    cv_results = cross_val_score(model, Xtrm, ytr, cv=10,scoring = 'neg_mean_squared_error') 
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 
    print(msg)

# Compara os algoritmos
sns.set(rc={'figure.figsize':(8,7)})
fig = plt.figure() 
fig.suptitle('Comparação de modelos') 
ax = fig.add_subplot(111) 
plt.boxplot(results) 
ax.set_xticklabels(names)
plt.show()
# Grid search: Treinamento para o melhor modelo Random Forest
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits=10)
model = RandomForestRegressor(random_state = 7)

# Grid search: parâmetros
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 15],
    'max_features': [2, 3, 4],
    #'min_samples_leaf': [3, 4, 5],
    #'min_samples_split': [8, 10],
    'n_estimators': [100, 200, 300]
}

# Execução do grid search
CV_model = GridSearchCV(estimator=model, param_grid=param_grid,cv=kfold,scoring=scoring)
CV_model_result = CV_model.fit(Xtrm, ytr)

# Print resultados
print("Best: %f using %s" % (CV_model_result.best_score_, CV_model_result.best_params_))
# Melhor modelo
baseline = RandomForestRegressor(**CV_model_result.best_params_)
baseline.fit(Xtrm,ytr)
# Previsão de casos
p = baseline.predict(Xvalm)
# Metricas
meanSquaredError=mean_squared_error(yval, p)
print("MSE:", round(meanSquaredError,2))
rootMeanSquaredError = sqrt(meanSquaredError)
print("RMSE:", round(rootMeanSquaredError,2))
# Avaliação da previsão 
a = pd.Series(p)
b = pd.Series(yval)
x = {'Previsto': a} 
y = {'Realizado': b} 
w = pd.DataFrame(x)
z = pd.DataFrame(y)

df_teste = pd.concat([w.reset_index(drop=True), z.reset_index()], axis=1)

df_teste[["Previsto","Realizado"]].plot(figsize=(15, 5),
                                        title='Gráfico de linhas - Previsto e Realizado',
                                        grid=True)
plt.show()

df_teste[["Previsto","Realizado"]].round(0).T
# Importancia das variaveis
sns.set(rc={'figure.figsize':(8, 8)})
features = Xtr.columns
importances = baseline.feature_importances_
indices = np.argsort(importances)

plt.title('Importancia das variáveis')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importancia relativa')
plt.show()