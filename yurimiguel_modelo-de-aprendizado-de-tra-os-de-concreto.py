#Importando as bibliotecas 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Criação de gráficos. 
import seaborn as sns # Desing de gráficos mais atraentes.

from scipy.stats import norm
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


%matplotlib inline
#Criando o DataFrame através do arquivo Concret_Data_Yeh
df = pd.read_csv('../input/Concrete_Data_Yeh.csv')

# Vamos Renomear as colunas com a tradução das nomenclaturas para o português.
# Todas as colunas de materiais estão na unidade de Kg/m³.
# A idade do concreto está em Dias.
# Já o Fck está em Mpa.
df.columns = ['Cimento','Escória de Alto Forno', 'Escória', 'Água', 'Superplastificante', 'Agregado Graudo', 'Agregado Fino', 'Idade do Concreto', 'Fck']
df.head(10)
# Vamos Arredondar o valor do Fck
df.round({'Fck': 0})
# Vamos agora obter uma descrição dos dados provenientes do DataFrame.
df.describe()
# Vamos avaliar a tipologia dos nossos dados;
df.dtypes
# Vamos plotar um gráfico de barras para determinar quantas unidade de corpor de prova temos 
# para cada idade do concreto;
df['Idade do Concreto'].value_counts().plot.bar()

plt.xlabel('Idade do Concreto')
plt.ylabel('Quantidade de ocorrências')
# Vamos criar um novo DataFrame apenas com os corpos de prova que tiveram sua resistência medida 
# com 28 dias

df28 = df[df['Idade do Concreto'] == 28]

df28.head()
# Agora vamos avaliar quantas entradas temos no nosso novo DataFrame
df28.info()
# Precisamos verificar a nova média do Fck para esse novo DataFrame
# uma vez que ela apresenta um valor de parâmetro de avaliação.
df28.describe()
# Vamos utilizar o PairPlot para tentar visualizar como é a distribuição do nossos dados.
sns.pairplot(df28)
sns.heatmap(df28.corr())
# Vamos Verificar se a variável de Target segue uma tendencia de distribuição normal.
sns.distplot(df28['Fck'], fit=norm);
fig = plt.figure()
res = stats.probplot(df28['Fck'], plot=plt)
# Vamos começar dividindo os nossos dados em atributos e rótulos

#Conjunto de todas as colunas do DataFrame,menos as colunas 'Fck' e 'Idade do Concreto'
X = df28.drop(['Fck', 'Idade do Concreto'], axis=1) 

#Conjunto apenas com a coluna 'Fck', variável que queremos prever
y = df28['Fck'] 
# Agora vamos dividir os nossos dados em conjuntos de treinamento e teste.
# Para esse exemplo vamos utilizar o método 'train_test_split', que fará a divisão automática dos dados.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Primeiramente vamos instanciar a classe 'DecisionForestRegressor' e chamar o método 'fit'
rf = RandomForestRegressor (n_estimators=1000, random_state=42)

rf.fit(X_train, y_train)
# Para fazer previsões no conjunto de teste vamos utilizar o método 'predict'
y_pred = rf.predict(X_test)
# Agora vamos comparar alguns dos nossos valores previstos com os valores reais e ver o quão preciso é o nosso modelo.
df_new = pd.DataFrame({'Valor_Real':y_test, 'Valor_Previsto':y_pred})
df_new
# Avaliando o modelo através do pacote 'metrics'.
print ('Erro Médio Absoluto:', metrics.mean_absolute_error(y_test, y_pred))
print ('Erro Quadrático Médio:', metrics.mean_squared_error(y_test, y_pred))
print ('Raiz do Erro Quadrático Médio:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Calculando o erro médio absoluto por outro meio
errors = abs(y_pred - y_test)

# Aqui calculamos o percentual do erro médio absoluto (MAPE)
mape =100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print ("Acurácia:", round(accuracy, 2), "%.")

