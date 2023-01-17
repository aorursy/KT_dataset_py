import pandas as pd

#biblioteca focado em estatística

from statsmodels.tsa.seasonal import seasonal_decompose

#bibliotesca de plotar gráficos

import matplotlib.pyplot as plt

import seaborn as sns

# Separar emtre treino e validação

from sklearn.model_selection import train_test_split

# Calcular o erro

from sklearn.metrics import mean_squared_error

sns.set_style()

import numpy as np



%matplotlib inline

#Configurar tamanhos dos gráficos como padrão para todos



%config InlineBackend.figure_format = 'svg'
path = "../input/consumo-eletricocsv/consumo_eletrico.csv"

dataset = pd.read_csv(path)

dataset.head(12)

# converter DATE para datetime e associar ao index do dataframe

dataset["DATE"] = pd.to_datetime(dataset.DATE, format="%m-%d-%Y")



dataset.set_index("DATE",inplace=True)

# ver as primeiras 5 entradas

dataset.head()
# Separar treino de test

X_train,X_test = train_test_split(dataset,shuffle=False,test_size=0.15)

# Copiar X_test para fazer pequenas previsões sem sujar o dataframe 

y_hat = X_test.copy()

# Verificar se foi realmente de forma sequencial a separação dos dados de acordo com os últimos números do X_test

display(dataset.tail())

display(X_test.tail())
# Copiar o último valor para o valor de treino 

y_hat['naive'] = X_train.iloc[-1].Value

y_hat.head()
#Plotar gráfico para montarmos um baseline

fig,ax = plt.subplots(figsize=(16,10))

X_train.plot(ax=ax)

X_test.plot(ax=ax)

# Como cada entrada possue uma cor, deixar o valor de naive constante fez com que observemos a parte que iriamos entender a parte que iriamos analizar 

y_hat['naive'].plot(ax=ax)

plt.show()
# Calcular erro do modelo de X_test usando a raiz quadrada

print("Erro do método naive")

mean_squared_error(y_hat.Value,y_hat.naive,squared = True)
#Vamos calcular a média móvel de 7 dias

y_hat["7_days"] = X_train.Value.rolling(7).mean()[-1]

y_hat.head()
#plotar novamente mas agora ao invés de plotar com a funçãao naive utilizaremos a média móvel de 7 dias

fig,ax = plt.subplots(figsize=(16,10))

X_train.plot(ax=ax)

X_test.plot(ax=ax)

# Como cada entrada possue uma cor, deixar o valor de naive constante fez com que observemos a parte que iriamos entender a parte que iriamos analizar 

y_hat['7_days'].plot(ax=ax)

plt.show()
# calcular o erro nesse caso

print('Erro ao usar média movel entre o intervalo de 7 dias')

mean_squared_error(y_hat.Value,y_hat["7_days"],squared = True)
#importando coisas do holt's linear

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.holtwinters import Holt
# colocar no gráfico usando a seasonal_decompose

result = seasonal_decompose(X_train)

fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(8,10))

result.observed.plot(ax=ax1)

result.trend.plot(ax=ax2)

result.seasonal.plot(ax=ax3)

result.resid.plot(ax=ax4)

plt.tight_layout()
# tabalhar a inclinação dos gráficos com o Holt

y_hat['holt'] = Holt(X_train.Value).fit(smoothing_level=0.1,smoothing_slope=0.1).forecast(len(X_test))

# plotar gráfico com holt

fig,ax = plt.subplots(figsize=(16,10))

X_train.plot(ax=ax)

X_test.plot(ax=ax)

# Como cada entrada possue uma cor, deixar o valor de naive constante fez com que observemos a parte que iriamos entender a parte que iriamos analizar 

y_hat['holt'].plot(ax=ax)

plt.show()
#Calcular o erro baseado utilizando o método holt

print("Erro ao utilizar o método holt")

mean_squared_error(X_test.Value,y_hat['holt'],squared=True)
## Importar test do dickey fuller

from statsmodels.tsa.stattools import adfuller

dataset = pd.read_csv(path,index_col=0,squeeze=True)

# Extrair somente os valores 

X = dataset.values

result = adfuller(X)

print('Teste Dickey-fuller')

display('Teste estatistíco é igual a: {:.4f}'.format(result[0]))

display('Valor p: {:.4f}'.format(result[1]))

display('Valores críticos: {:.4f}'.format(result[2]))



for key,value in result[4].items():

    print('{}: {:.4f}'.format(key,value))

    
# media móvel de 12 dias

media_movel_12 = dataset.rolling(12).mean()

fig,ax = plt.subplots()

dataset.plot(ax=ax,legend=False)

media_movel_12.plot(ax=ax,legend=False,color='r')

plt.tight_layout()

# Exemplo da aplicabilidade sem o log em uma função de primeiro grau

a = np.arange(0,10000)

plt.plot(a)
# Exemplo da aplicabilidade com o log em uma função de primeiro grau

b = np.log(a)

plt.plot(a,b)
# Aplicando o log no nosso dataset de série temporal

df_log = np.log(dataset)

media_movel_log = df_log.rolling(12).mean()

fig, ax = plt.subplots()

df_log.plot(ax=ax)

media_movel_log.plot(ax=ax)

plt.tight_layout()