import pandas as pd

#biblioteca focado em estatística

from statsmodels.tsa.seasonal import seasonal_decompose

#bibliotesca de plotar gráficos

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style()



%matplotlib inline

#Configurar tamanhos dos gráficos como padrão para todos

%config InlineBackend.rc={'figure.figsize': (15, 10)}
# Importando dataset

df_trem = pd.read_csv("../input/train-data/trem.csv")

df_trem.head(10)
# Converter para datetime

df_trem.Datetime = pd.to_datetime(df_trem.Datetime,format="%d-%m-%Y %H:%M")

# plotar relação entre tempo e quantidade de pessoas

plt.plot(df_trem.Datetime,df_trem.Count);
# Criar novas features a partir de uma que já existe



df_trem['day'] = df_trem.Datetime.dt.day

df_trem['month'] = df_trem.Datetime.dt.month

df_trem['year'] = df_trem.Datetime.dt.year

df_trem['hour'] = df_trem.Datetime.dt.hour

df_trem['day_of_week'] = df_trem.Datetime.dt.dayofweek

# Mudar dados para 0 ou 1 para ficar mais acertivo

df_trem['weekend'] = 0

df_trem.loc[(df_trem.day_of_week == 5) | (df_trem.day_of_week == 6),"weekend"] = 1

df_trem.head()
# Verificar a se as pessoas saem mais no fim de semana ou na semana sendo 0 dia de semana e 1 final de semana

df_trem['weekend'].value_counts()

#plotar gráfico de barra

df_trem.groupby('weekend').Count.mean().plot.bar();
# Verificar os horários de pico

df_trem.groupby('hour').Count.mean().plot.bar();
# Verificar mês que mais incidência de pessoas utilizando o metrô

df_trem.groupby('month').Count.mean().plot.bar();
# Verificar dia da semana que mais tem pessoas utilizando o metrô

df_trem.groupby('day_of_week').Count.mean().plot.bar();
# Verificar o ano com maior incidência de utilização do metrô

df_trem.groupby('year').Count.mean().plot.bar();