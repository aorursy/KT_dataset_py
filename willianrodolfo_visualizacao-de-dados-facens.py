# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Jefferson Rodrigo de Almeida - Ra: 183499

# Willian Rodolfo de ALmeida - Ra: 183502
# TRABALHO DE VISUALIZAÇÃO DE DADOS

# DATAFRAME FLIGHTS
import pandas as pd

import numpy as np

import datetime, warnings, scipy

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvas

from matplotlib.figure import Figure

from IPython.display import display_png

%matplotlib inline
df_voo = pd.read_csv('../input/flights.csv')
#VERIFICANDO AS INFORMAÇÕES DO DATAFRAME
df_voo.info()
#FUNÇÃO PARA CONVERTER TEMPO
def format_heure(chaine):

    if pd.isnull(chaine):

        return np.nan

    else:

        if chaine == 2400: chaine = 0

        chaine = "{0:04d}".format(int(chaine))

        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))

        return heure
df_voo['dep_time'] = df_voo['dep_time'].apply(format_heure)

df_voo['sched_dep_time'] = df_voo['sched_dep_time'].apply(format_heure)

df_voo['sched_arr_time'] = df_voo['sched_arr_time'].apply(format_heure)

df_voo['arr_time'] = df_voo['arr_time'].apply(format_heure)
df_voo.head(2)
#VERIFICANDO A QUANTIDADE DE VALORES NULOS NO DATAFRAME
df_voo.isnull().sum()
df_voo.dropna(inplace=True)
#REMOVENDO OS VALORES NULOS.
df_voo.isnull().sum()
df_voo.head(2)
#NOVO DATAFRAME COM OS DADOS DOS AEROPORTOS
df_aeroportos = pd.read_csv('../input/airports.csv')
df_aeroportos.head(10)
#NOVA DATAFRAME DAS EMPRESAS AÉREAS
df_EmpresasAereas = pd.read_csv('../input/airlines.csv')

df_EmpresasAereas.head(18)
# TOTAL DE VOOS POR COMPANHIA AÉREA
df_Emp_Total = pd.DataFrame(df_voo['carrier'].value_counts())

df_Emp_Total.columns = ['total']

df_Emp_Total.head(16)
#GRÁFICO COM O TOTAL DE VOOS POR COMPANHIA AÉREA    
plt.figure(figsize=(10,5))

for x, y in df_Emp_Total['total'].items():

    plt.bar(x,y)

    plt.text(x, y, s=y, horizontalalignment='center')



plt.xlabel('Empresas')

plt.ylabel('Voos')

plt.title('Empresas Aéreas')

plt.legend(['United Air Lines Inc.', 'JetBlue Airways', 'Atlantic Southeast Airlines', 'Delta Air Lines Inc.',

            'American Airlines Inc.','American Eagle Airlines Inc.', 'US Airways Inc.','9E','Southwest Airlines Co.',

            'Virgin America', 'FL', 'Alaska Airlines Inc.', 'Frontier Airlines Inc.', 'YV','Hawaiian Airlines Inc.','Skywest Airlines Inc.'

           ],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
#TOP 3 COMPANHIA AÉREA

#DECIDIMOS SEPARAR AS 3 MAIORES EMPRESAS POR TER MAIOR NUMERO DE VOOS.
plt.figure(figsize=(4,7))



for x, y in df_Emp_Total['total'].head(3).items():

    plt.bar(x,y)

    plt.text(x, y, s=y, horizontalalignment='center')





plt.xlabel('Empresas')

plt.ylabel('Voos')

plt.title('Empresas Aéreas')

plt.legend(['United Air Lines Inc', 'JetBlue Airways', 'Atlantic Southeast Airlines'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
#COMPANHIA AÉREA UNITED AIR LINES INC
df_voo.head()
#AEROPORTOS COM DECOLAGENS DA COMPANHIA UNITED AIR LINES INC
df_UA = pd.DataFrame(df_voo.groupby(by=['carrier','origin']).size()['UA'], columns=['Qtd'])

df_UA = df_UA.merge(df_aeroportos, left_on='origin', right_on='IATA_CODE')

df_UA.head()
#GRÁFICO REPRESENTANDO OS 3 AEROPORTOS COM MAIOR QUANTIDADE DE VOOS DA COMPANHIA


labels = []

sizes = []

explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

for x in range(len(df_UA)):

    sizes.append(df_UA['Qtd'][x])

    labels.append(df_UA['IATA_CODE'][x])

    

ax1.pie(sizes,  explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.xlabel('AEROPORTOS')

plt.title('United Air Lines Inc - (UA) - Origem')

plt.legend(['Newark Liberty International Airport - Qtd: 45501', 'John F. Kennedy International AirportÂ (New York International Airport - Qtd: 4478', 'LaGuardia Airport (Marine Air Terminal) - Qtd: 7803'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()



plt.show()

df = pd.DataFrame(df_voo.groupby(by=['carrier','origin','month']).size()['UA']['EWR'], columns=['Total'])
#CRIANDO UM DATAFRAME PARA SEPARAÇÃO DOS VOOS POR MESES.
df_Mes = pd.DataFrame(['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],columns=['Mes_desc'])



df_Mes['Mes_num'] =[1,2,3,4,5,6,7,8,9,10,11,12] 

df_Mes.head()
df = df.merge(df_Mes, left_on='month', right_on='Mes_num')

df.head()
#AEROPORTO COM MAIS VOOS É NEWARK LIBERTY INTERNATIONAL AIRPORT, ABAIXO O DESMEMBRAMENTO DO ANO EM MESES
plt.figure(figsize=(10,5))

for x in df['Mes_num'] -1:

    plt.bar(df['Mes_desc'][x],df['Total'][x])

    plt.text(df['Mes_desc'][x],df['Total'][x], s=df['Total'][x], horizontalalignment='center')

     



plt.xlabel('Mês')

plt.ylabel('Origem Voo')

plt.title('Newark Liberty International Airport - United Air Lines Inc - (UA)')

plt.show()
#PARA GERAR UM HISTOGRAMA DE DECOLAGEM UTILIZAMOS O MES DE AGOSTO COMO EXEMPLO

#PARA ENCONTRAR O VALOR DE BINS FOI UTILIZADO A FÓRMULA: k = 1 + 3,322(log10 n)
df_Ago_UA = df_voo.loc[(df_voo['carrier']) == 'UA']

df_Ago_UA = df_Ago_UA.loc[(df_voo['month']) == 8,['dep_delay','origin']]

df_Ago_UA = df_Ago_UA.loc[(df_voo['origin']) == 'EWR']



df_Ago_UA['dep_delay'].plot.hist(bins=13, alpha=0.5, density=True);df_Ago_UA['dep_delay'].plot.kde()

    
#GRÁFICO DE VIOLINO MOSTRANDO A QUANTIDADE DE DADOS APRESENTADA NO HISTOGRAMA
df_Ago_UA.head()
plt.figure(figsize=(7, 6))

sns.set_style('whitegrid')

sns.violinplot(x='origin', y='dep_delay', cut=0, scale="count", data=df_Ago_UA)
#CRIANDO UM DATAFRAME COM AS INFORMAÇÕES DE DESTINO DOS VOOS
df_UA_Destino = pd.DataFrame(df_voo.groupby(by=['carrier','dest']).size()['UA'], columns=['Qtd'])

df_UA_Destino = df_UA_Destino.sort_values(['Qtd'], ascending=False)

df_UA_Destino.head()
#AEROPORTOS DE ATERRISSAGEM DA EMPRESA UNITED AIR LINES INC
plt.figure(figsize=(10,5))

for x, y in df_UA_Destino['Qtd'].items():

    plt.bar(x,y)

    plt.text(x, y, s=y, horizontalalignment='center', verticalalignment='center')



plt.xlabel('Aeroportos')

plt.ylabel('Voos')

plt.title('Aeroportos de Destinos')

plt.show()
#SEPARAMOS AS TOP 5 DOS AEROPORTOS POR SEREM MUITOS
df_UA_Destino['Qtd'].head(5)
#GRÁFICO COM OS 5 AEROPORTOS COM MAIS VOOS DESSA COMPANHIA


labels = []

sizes = []

explode = (0.1, 0, 0, 0 ,0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

for x, y in df_UA_Destino['Qtd'].head(5).items():

    sizes.append(y)

    labels.append(x)

    

ax1.pie(sizes,   explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.xlabel('AEROPORTOS')

plt.title('United Air Lines Inc - (UA) - Destino')

plt.legend(['George Bush Intercontinental Airport - Qtd = 6814', 'Chicago O Hare International Airport - Qtd = 6744', 

           'San Francisco International Airport - Qtd = 6728', 'Los Angeles International Airport - Qtd = 5770','Denver International Airport - Qtd = 3737'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
# CRIANDO DATAFRAME PARA O AEROPORTO COM MAIOR NÚMERO DE VOOS E SEPARANDO POR MÊS
df_destin = pd.DataFrame(df_voo.groupby(by=['carrier','dest','month']).size()['UA']['IAH'], columns=['Total'])
df_destin = df_destin.merge(df_Mes, left_on='month', right_on='Mes_num')

df_destin.head()
#GRÁFICO COM A QUANTIDADE DE VOOS RECEBIDOS PELO AEROPORTO IAH
plt.figure(figsize=(10,5))

for x in df['Mes_num'] -1:

    plt.bar(df['Mes_desc'][x],df['Total'][x])

    plt.text(df['Mes_desc'][x],df['Total'][x], s=df['Total'][x], horizontalalignment='center')

     



plt.xlabel('Mês')

plt.ylabel('Destino Voo')

plt.title('George Bush Intercontinental Airport - United Air Lines Inc - (UA)')

plt.show()
# CRIANDO UM DATAFRAME PARA UTILIZAR O MÊS DE AGOSTO COMO EXEMPLO PARA ANÁLISE DOS DADOS
df_Ago_IAH = df_voo.loc[(df_voo['carrier']) == 'UA']

df_Ago_IAH.head()
len(df_Ago_IAH)
#GERADO UM HISTOGRAMA DE ATERRISSAGEM - MÊS DE AGOSTO 

#PARA ENCONTRAR O VALOR DE BINS FOI UTILIZADO A FÓRMULA: k = 1 + 3,322(log10 n)
df_Ago_IAH = df_Ago_IAH.loc[(df_voo['month']) == 8,['dep_delay','dest']]

df_Ago_IAH = df_Ago_IAH.loc[(df_voo['dest']) == 'IAH']



df_Ago_IAH['dep_delay'].plot.hist(bins=10, alpha=0.5, density=True);df_Ago_IAH['dep_delay'].plot.kde()



df_Ago_IAH.head()
#GRÁFICO DE VIOLINO MOSTRANDO A QUANTIDADE DE DADOS APRESENTADA NO HISTOGRAMA
plt.figure(figsize=(7, 6))

sns.set_style('whitegrid')

sns.violinplot(x='dest', y='dep_delay', cut=0, scale="count", data=df_Ago_IAH)
#COMPANHIA AÉREA B6 - JETBLUE AIRWAYS
#CRIANDO DATAFRAME PARA A EMPRESA B6
df_B6 = pd.DataFrame(df_voo.groupby(by=['carrier','origin']).size()['B6'], columns=['Qtd'])

df_B6 = df_B6.merge(df_aeroportos, left_on='origin', right_on='IATA_CODE')



df_B6.head()

#GRÁFICO REPRESENTANDO OS 3 AEROPORTOS COM MAIS VOOS DESSA COMPANHIA
labels = []

sizes = []

explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

for x in range(len(df_B6)):

    sizes.append(df_B6['Qtd'][x])

    labels.append(df_B6['IATA_CODE'][x])

    

ax1.pie(sizes,  explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.xlabel('AEROPORTOS')

plt.title('JetBlue Airways – B6 - Origem')

plt.legend(['Newark Liberty International Airport - Qtd: 6472', 'John F. Kennedy International AirportÂ (New York International Airport - Qtd: 41666', 'LaGuardia Airport (Marine Air Terminal) - Qtd: 5911'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()



plt.show()
# DATAFRAME PARA SEPARAÇÃO POR MÊS
df_B6_Month = pd.DataFrame(df_voo.groupby(by=['carrier','origin','month']).size()['B6']['JFK'], columns=['Total'])
df_B6_Month.head()
df_B6_Month = df_B6_Month.merge(df_Mes, left_on='month', right_on='Mes_num')

df_B6_Month.head()
#GRÁFICO DO AEROPORTO COM A MAIOR QUANTIDADE DE VOOS POR MÊS 
plt.figure(figsize=(10,5))

for x in df_B6_Month['Mes_num'] -1:

    plt.bar(df_B6_Month['Mes_desc'][x],df['Total'][x])

    plt.text(df_B6_Month['Mes_desc'][x],df['Total'][x], s=df_B6_Month['Total'][x], horizontalalignment='center')

     



plt.xlabel('Mês')

plt.ylabel('Origem Voo')

plt.title('John F. Kennedy International Airport (New York International Airport) - Qtd: 41666')

plt.show()
#GERADO UM HISTOGRAMA DE DECOLAGEM - MÊS DE AGOSTO 

#PARA ENCONTRAR O VALOR DE BINS FOI UTILIZADO A FÓRMULA: k = 1 + 3,322(log10 n)
df_Ago_b6 = df_voo.loc[(df_voo['carrier']) == 'B6']

df_Ago_b6 = df_Ago_b6.loc[(df_voo['month']) == 8,['dep_delay','origin']]

df_Ago_b6 = df_Ago_b6.loc[(df_voo['origin']) == 'JFK']



df_Ago_b6['dep_delay'].plot.hist(bins=13, alpha=0.5, density=True);df_Ago_b6['dep_delay'].plot.kde()

    
#GRÁFICO DE VIOLINO MOSTRANDO A QUANTIDADE DE DADOS APRESENTADA NO HISTOGRAMA
plt.figure(figsize=(7, 6))

sns.set_style('whitegrid')

sns.violinplot(x='origin', y='dep_delay', cut=0, scale="count", data=df_Ago_b6)
# DATAFRAME COM OS AEROPORTOS DE DESTINO SAINDO DE John F. Kennedy International Airport (New York International Airport)
df_B6_Destino = pd.DataFrame(df_voo.groupby(by=['carrier','dest']).size()['B6'], columns=['Qtd'])

df_B6_Destino = df_B6_Destino.sort_values(['Qtd'], ascending=False)

df_B6_Destino.head()
#GRÁFICO COM OS 5 AEROPORTOS COM MAIOR NÚMERO DE ATERRISSAGEM
labels = []

sizes = []

explode = (0.1, 0, 0, 0 ,0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

for x, y in df_B6_Destino['Qtd'].head(5).items():

    sizes.append(y)

    labels.append(x)

    

ax1.pie(sizes,   explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.xlabel('AEROPORTOS')

plt.title('JetBlue Airways – B6 - Destino')

plt.legend(['Fort Lauderdale-Hollywood International Airport - Qtd: 6466',

            'Orlando International Airport - Qtd: 6409',

            'Gen. Edward Lawrence Logan International Airport - Qtd: 4325',

            'Palm Beach International Airport - Qtd: 3126',

            'Buffalo Niagara International Airport - Qtd: 2773'

           ],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
#DATAFRAME SEPARANDO POR MÊS O AEROPORTO COM MAIOR NÚMERO DE ATERRISSAGEM
df_destin_b6 = pd.DataFrame(df_voo.groupby(by=['carrier','dest','month']).size()['B6']['FLL'], columns=['Total'])
df_destin_b6 = df_destin_b6.merge(df_Mes, left_on='month', right_on='Mes_num')
#GRÁFICO COM A QUANTIDADE POR MÊS
plt.figure(figsize=(10,5))

for x in df_destin_b6['Mes_num'] -1:

    plt.bar(df_destin_b6['Mes_desc'][x],df_destin_b6['Total'][x])

    plt.text(df_destin_b6['Mes_desc'][x],df_destin_b6['Total'][x], s=df_destin_b6['Total'][x], horizontalalignment='center')

     



plt.xlabel('Mês')

plt.ylabel('Destino Voo')

plt.title('Fort Lauderdale-Hollywood International Airport - Qtd: 6466')

plt.show()
#GERADO UM HISTOGRAMA DE ATERRISSAGEM - MÊS DE AGOSTO 

#PARA ENCONTRAR O VALOR DE BINS FOI UTILIZADO A FÓRMULA: k = 1 + 3,322(log10 n)
df_Ago_b6_dest = df_voo.loc[(df_voo['carrier']) == 'B6']

df_Ago_b6_dest = df_Ago_b6_dest.loc[(df_voo['month']) == 8,['dep_delay','dest']]

df_Ago_b6_dest = df_Ago_b6_dest.loc[(df_voo['dest']) == 'FLL']

len(df_Ago_b6_dest)
df_Ago_b6_dest['dep_delay'].plot.hist(bins=10, alpha=0.5, density=True);df_Ago_b6_dest['dep_delay'].plot.kde()
#GRÁFICO DE VIOLINO MOSTRANDO A QUANTIDADE DE DADOS APRESENTADA NO HISTOGRAMA
plt.figure(figsize=(7, 6))

sns.set_style('whitegrid')

sns.violinplot(x='dest', y='dep_delay', cut=0, scale="count", data=df_Ago_b6_dest)
#COMPANHIA AÉREA EV - ATLANTIC SOUTHEAST AIRLINES
#DATAFRAME PARA A COMPANHIA AÉREA EV
df_EV = pd.DataFrame(df_voo.groupby(by=['carrier','origin']).size()['EV'], columns=['Qtd'])

df_EV = df_EV.merge(df_aeroportos, left_on='origin', right_on='IATA_CODE')



df_EV.head()
#GRÁFICO PARA OS AEROPORTOS COM MAIOR NÚMERO DE DECOLAGENS 
labels = []

sizes = []

explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

for x in range(len(df_EV)):

    sizes.append(df_EV['Qtd'][x])

    labels.append(df_EV['IATA_CODE'][x])

    

ax1.pie(sizes,  explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.xlabel('AEROPORTOS')

plt.title('Atlantic Southeast Airlines- EV - Origem')

plt.legend(['Newark Liberty International Airport - Qtd: 41557', 'John F. Kennedy International AirportÂ (New York International Airport - Qtd: 1326', 'LaGuardia Airport (Marine Air Terminal) - Qtd: 8225'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()



plt.show()
#DATAFRAME COM A SEPRAÇÃO POR MÊS DO AEROPORTO COM MAIOR NÚMERO DE VOOS
df_EV_Month = pd.DataFrame(df_voo.groupby(by=['carrier','origin','month']).size()['EV']['EWR'], columns=['Total'])

df_EV_Month = df_EV_Month.merge(df_Mes, left_on='month', right_on='Mes_num')

df_EV_Month.head()
#GRÁFICO COM A QUANTIDADE DE VOOS POR MÊS
plt.figure(figsize=(10,5))

for x in df_EV_Month['Mes_num'] -1:

    plt.bar(df_EV_Month['Mes_desc'][x],df['Total'][x])

    plt.text(df_EV_Month['Mes_desc'][x],df['Total'][x], s=df_EV_Month['Total'][x], horizontalalignment='center')

     



plt.xlabel('Mês')

plt.ylabel('Origem Voo')

plt.title('Newark Liberty International Airport - Qtd: 41557')

plt.show()
#GERADO UM HISTOGRAMA DE DECOLAGEM - MÊS DE AGOSTO 

#PARA ENCONTRAR O VALOR DE BINS FOI UTILIZADO A FÓRMULA: k = 1 + 3,322(log10 n)
df_Ago_EV_ORIGIN = df_voo.loc[(df_voo['carrier']) == 'EV']

df_Ago_EV_ORIGIN = df_Ago_EV_ORIGIN.loc[(df_voo['month']) == 8,['dep_delay','origin']]

df_Ago_EV_ORIGIN = df_Ago_EV_ORIGIN.loc[(df_voo['origin']) == 'EWR']

len(df_Ago_EV_ORIGIN)
df_Ago_EV_ORIGIN['dep_delay'].plot.hist(bins=13, alpha=0.5, density=True);df_Ago_EV_ORIGIN['dep_delay'].plot.kde()
#GRÁFICO DE VIOLINO MOSTRANDO A QUANTIDADE DE DADOS APRESENTADA NO HISTOGRAMA
plt.figure(figsize=(7, 6))

sns.set_style('whitegrid')

sns.violinplot(x='origin', y='dep_delay', cut=0, scale="count", data=df_Ago_EV_ORIGIN)
#DATAFRAME PARA OS DESTINOS COM MAIOR NÚMERO DE ATERRISSAGEM
df_ev_Destino = pd.DataFrame(df_voo.groupby(by=['carrier','dest']).size()['EV'], columns=['Qtd'])

df_ev_Destino = df_ev_Destino.sort_values(['Qtd'], ascending=False)

df_ev_Destino.head()
#GRÁFICO COM OS 5 AEROPORTOS COM MAIS ATERRISSAGEM 
labels = []

sizes = []

explode = (0.1, 0, 0, 0 ,0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

for x, y in df_ev_Destino['Qtd'].head(5).items():

    sizes.append(y)

    labels.append(x)

    

ax1.pie(sizes,   explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.xlabel('AEROPORTOS')

plt.title('Atlantic Southeast Airlines- EV - Destino')

plt.legend(['Washington Dulles International Airport - Qtd: 3824',

            'Detroit Metropolitan Airport - Qtd: 2389',

            'Charlotte Douglas International Airport - Qtd: 2374',

            'Nashville International Airport - Qtd: 2054',

            'Richmond International Airport - Qtd: 2025'

           ],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
#DATAFRAME PARA SEPARAÇÃO POR MÊS E A QUANTIDADE DE CADA MÊS
df_destin_ev = pd.DataFrame(df_voo.groupby(by=['carrier','dest','month']).size()['EV']['IAD'], columns=['Total'])

df_destin_ev = df_destin_ev.merge(df_Mes, left_on='month', right_on='Mes_num')



plt.figure(figsize=(10,5))

for x in df_destin_ev['Mes_num'] -1:

    plt.bar(df_destin_ev['Mes_desc'][x],df_destin_ev['Total'][x])

    plt.text(df_destin_ev['Mes_desc'][x],df_destin_ev['Total'][x], s=df_destin_ev['Total'][x], horizontalalignment='center')

     



plt.xlabel('Mês')

plt.ylabel('Destino Voo')

plt.title('Washington Dulles International Airport - Qtd: 3824')

plt.show()
#GERADO UM HISTOGRAMA DE ATERRISSAGEM - MÊS DE AGOSTO 

#PARA ENCONTRAR O VALOR DE BINS FOI UTILIZADO A FÓRMULA: k = 1 + 3,322(log10 n)
df_Ago_EV_dest = df_voo.loc[(df_voo['carrier']) == 'EV']

df_Ago_EV_dest = df_Ago_EV_dest.loc[(df_voo['month']) == 8,['dep_delay','dest']]

df_Ago_EV_dest = df_Ago_EV_dest.loc[(df_voo['dest']) == 'IAD']

len(df_Ago_EV_dest)
df_Ago_EV_dest['dep_delay'].plot.hist(bins=9, alpha=0.5, density=True);df_Ago_EV_dest['dep_delay'].plot.kde()
#GRÁFICO DE VIOLINO MOSTRANDO A QUANTIDADE DE DADOS APRESENTADA NO HISTOGRAMA
plt.figure(figsize=(7, 6))

sns.set_style('whitegrid')

sns.violinplot(x='dest', y='dep_delay', cut=0, scale="count", data=df_Ago_EV_dest)
# INFORMAÇÕES DOS 3 DATAFRAMES DAS COMPANHIAS AÉREAS UTILIZADOS
df_UA.head()
df_B6.head()
df_EV.head()
# DATAFRAME COM AS INFORMAÇÕES DOS 3 DATAFRAMES
index = ['EWR', 'JFK', 'LGA']

df_3_Emp = pd.DataFrame({'UA' : [45501, 4478, 7803],

                         'B6' : [6472, 41666, 5911],

                         'EV' : [41557, 1326, 8225]}, index = index)

df_3_Emp.head()
sns.factorplot('IATA_CODE','Qtd', data=df_UA)

sns.factorplot('IATA_CODE','Qtd', data=df_B6)

sns.factorplot('IATA_CODE','Qtd', data=df_EV)
#GRÁFICO DE CORRELAÇÃO ENTRE AS 3 COMPANHIAS AÉREAS
corr = df_3_Emp.corr()

corr.style.background_gradient()
sns.heatmap(corr,  cmap='RdBu', square=True, linewidths=.5)
sns.pairplot(df_3_Emp)