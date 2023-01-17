import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        


# Vai ao Dataset da Jonh Hopkins e lê a última versão

import pandas as pd
baseURL = "/kaggle/input/novel-corona-virus-2019-dataset/"
def loadData(fileName, columnName): 
    data = pd.read_csv(baseURL + fileName) \
             .drop(['Lat', 'Long'], axis=1) \
             .melt(id_vars=['Province/State', 'Country/Region'], var_name='date', value_name=columnName) \
             .fillna('<all>')
    data['date'] = data['date'].astype('datetime64[ns]')
    return data

allData = loadData("time_series_covid_19_confirmed.csv", "CumConfirmed") \
    .merge(loadData("time_series_covid_19_deaths.csv", "CumDeaths")) \
    .merge(loadData("time_series_covid_19_recovered.csv", "CumRecovered"))
allData = allData[pd.to_numeric(allData['CumConfirmed'], errors='coerce').notnull()]
allData = allData[pd.to_numeric(allData['CumDeaths'], errors='coerce').notnull()]
allData = allData[pd.to_numeric(allData['CumRecovered'], errors='coerce').notnull()]
countries = allData['Country/Region'].unique()
countries.sort()
allData.tail()

# Vai ao ficheiro excel buscar a versão anterior. Este ficheiro Excel pode ser editado para coorigir dados ou acrescentar
# dados à mão, como o último dia 
anterior = pd.read_excel("/kaggle/input/bd-hopkins-corrected-and-updated/bdHopkins.xlsx")
anterior.tail()
# Concatena novos os dados com os anteriores. Os dados anteriores prevalecem sempre para a data/país/província

novo = pd.concat([anterior, allData])
novo.drop_duplicates(subset=['date', 'Country/Region', 'Province/State'], inplace=True, keep='first')

# Reescreve o ficheiro bdHopkins com os dados novos acrescentados, mas sem estragar o que lá estava antes
# Por isso o ficheiro bdHopkins pode ser editado à mão, porque as correções e os acrescentos prevalecem
novo.to_excel("bdHopkins.xlsx", index = False)
novo
ultimaData = anterior[anterior["Country/Region"] == "Portugal"]["date"].max()
ultimoDia = ultimaData.date()
print("Última data em que há dados de Portugal: ", ultimoDia)

print("\033[91m #-------------------------------#")
print("\033[91m #-------------------------------#")
print("\033[91m #-------------------------------#")
ultimaPortugalCasos = anterior[anterior["Country/Region"] == "Portugal"]["CumConfirmed"].max()
print("\033[91m Última num de casos em Portugal: \033[1m", ultimaPortugalCasos)

ultimaIndiaCasos = anterior[anterior["Country/Region"] == "India"]["CumConfirmed"].max()
print("\033[93m Última num de casos na india: \033[1m", ultimaIndiaCasos)

print("\033[91m #-------------------------------#")
print("\033[91m #-------------------------------#")
print("\033[91m #-------------------------------#")
#Vars
diasMax = 30
casosMax = 20000

# Constrói o gráfico comparativo da evolução nos 20 dias a partir dos 100 casos
import numpy as np
tabela = pd.pivot_table(novo, values='CumConfirmed', index=['date', 'Country/Region'], aggfunc=np.sum)
tabela.reset_index(inplace = True)
tabela['Casos'] = tabela.CumConfirmed.astype('int32')
tabela['Data'] = pd.to_datetime(tabela['date'])
del tabela['date']
del tabela['CumConfirmed']
tabela = tabela[tabela['Casos'] > 100]

#Vitor Limit the top for clarity in low numbers
tabela = tabela[tabela['Casos'] < casosMax]

tabela.rename(columns = {'Country/Region' : 'País'}, inplace = True)
paises = tabela['País'].unique()
tudo = pd.DataFrame()
for pais in paises:
    pais = tabela[tabela['País'] == pais]
    pais.reset_index(inplace = True, drop = True)
    pais.reset_index(inplace = True)
    tudo = pd.concat([tudo, pais], axis = 0)
tudo.rename(columns = {'index' : 'Dias'}, inplace = True)
tudo = tudo[tudo['País'] != 'Others']
tudo1 = tudo[tudo['País'] != 'China']

selecao = ['Italy', 'Spain',  'Belgium', 'Turkey', 'Korea, South', 'Portugal', 'India']
tudo2 = tudo1[tudo1['País'].isin(selecao)]
tabela2 = pd.pivot_table(tudo2, values='Casos', index=['Dias'],
                    columns=['País'], aggfunc=np.sum)
tabela2 = tabela2[tabela2.index <= diasMax]
resumo = tabela2

import matplotlib.pyplot as plt 
%matplotlib inline
#plt.figure()
resumo.plot(style = {'Turkey': 'm--', 'Spain' : 'b--', 'Italy' : 'g--', 'Belgium' : 'y--',\
            'Korea, South': 'm-d', 'Portugal' : 'b-d', 'India':'g-d'}, \
            title = 'Casos nos '+str(diasMax)+' dias a seguir ao caso 100 até '+str(casosMax)+' casos - situação em: {}'.format(ultimoDia), \
            figsize=(20,10));
plt.grid(which="both", axis="x")
plt.xlabel('Dias desde o centésimo caso')
plt.ylabel('Número de Casos')

# V -sorting labalsmanually
# get current handles and labels
# this must be done AFTER plotting
current_handles, current_labels = plt.gca().get_legend_handles_labels()
# -sort or reorder the labels and handles
# reversed_handles = list(reversed(current_handles))
# reversed_labels = list(reversed(current_labels))
#print("current_handles: ")
#print(*current_handles, sep = "\n") 
#print("\n", )
# print(current_labels)
# print(*current_labels, sep = "\n") 
# print("print done", )
# -call plt.legend() with the new values
# plt.legend(reversed_handles,reversed_labels)
# plt.show()
# nao está a funcionar nada disto
#oredered_handles = ['Line2D(Spain)', 'Line2D(Italy)', 'Line2D(Germany)', 'Line2D(France)', 'Line2D(Korea, South)', 'Line2D(Portugal)', 'Line2D(India)']
#oredered_labels = ['Spain', 'Italy', 'Germany', 'France', 'Korea, South', 'Portugal', 'India']

#plt.legend(oredered_handles,oredered_labels)

ultimaPortugalCasos = anterior[anterior["Country/Region"] == "Portugal"]["CumConfirmed"].max()
print("Última num de casos em Portugal: \033[1m", ultimaPortugalCasos,"\033[0m")

ultimaIndiaCasos = anterior[anterior["Country/Region"] == "India"]["CumConfirmed"].max()
print("Última num de casos na india: \033[1m", ultimaIndiaCasos,"\033[0m (com +/- 1 dia de atraso)") 

# Constrói o gráfico comparativo da evolução nos 20 dias a partir dos 100 casos
import numpy as np
tabela = pd.pivot_table(novo, values='CumConfirmed', index=['date', 'Country/Region'], aggfunc=np.sum)
tabela.reset_index(inplace = True)
tabela['Casos'] = tabela.CumConfirmed.astype('int32')
tabela['Data'] = pd.to_datetime(tabela['date'])
del tabela['date']
del tabela['CumConfirmed']
tabela = tabela[tabela['Casos'] > 100]

tabela.rename(columns = {'Country/Region' : 'País'}, inplace = True)
paises = tabela['País'].unique()
tudo = pd.DataFrame()
for pais in paises:
    pais = tabela[tabela['País'] == pais]
    pais.reset_index(inplace = True, drop = True)
    pais.reset_index(inplace = True)
    tudo = pd.concat([tudo, pais], axis = 0)
tudo.rename(columns = {'index' : 'Dias'}, inplace = True)
tudo.tail()
tudo = tudo[tudo['País'] != 'Others']
tudo1 = tudo[tudo['País'] != 'China']

selecao = ['Italy', 'Spain',  'Belgium', 'Turkey', 'Korea, South', 'Portugal', 'India']
tudo2 = tudo1[tudo1['País'].isin(selecao)]
tabela2 = pd.pivot_table(tudo2, values='Casos', index=['Dias'],
                    columns=['País'], aggfunc=np.sum)
resumo = tabela2

import matplotlib.pyplot as plt 
%matplotlib inline
#plt.figure()
resumo.plot(style = {'Turkey': 'm--', 'Spain' : 'b--', 'Italy' : 'g--', 'Belgium' : 'y--',\
            'Korea, South': 'm-d', 'Portugal' : 'b-d', 'India':'g-d'}, \
            title = 'Casos nos 20 dias a seguir ao caso 100 - situação em: {}'.format(ultimoDia), \
            figsize=(20,10));
plt.xlabel('Dias desde o centésimo caso')
plt.ylabel('Número de Casos')

ultimaPortugalCasos = anterior[anterior["Country/Region"] == "Portugal"]["CumConfirmed"].max()
print("Última num de casos em Portugal: \033[1m", ultimaPortugalCasos,"\033[0m")

ultimaIndiaCasos = anterior[anterior["Country/Region"] == "India"]["CumConfirmed"].max()
print("Última num de casos na india: \033[1m", ultimaIndiaCasos,"\033[0m (com +/- 1 dia de atraso)") 
#anterior["Country/Region"].unique()

#percentual = resumo.pct_change()[1:]*100
#percentual = percentual[percentual["Portugal"] != 0 ]
#plt.figure()
#percentual.plot(style = {'Portugal' : 'k'}, \
#            title = 'Variação percentual de casos nos 20 dias a seguir ao caso 100 - situação em: {}'.format(ultimoDia), figsize=(20,10));
#percentual.plot(style = {'Portugal' : 'k'}, \
#            title = 'Variação percentual de casos nos 20 dias a seguir ao caso 100 - situação em: {}'.format(ultimoDia), \
#            figsize=(20,10), kind='bar');
#plt.xlabel('Dias desde o centésimo caso')
#plt.ylabel('Variação percentual')
#percentual2 = percentual
#percentual2["Média"] = percentual2.mean(axis = 1)
#compara2 = percentual2[['Portugal', 'Média']]
#print(selecao)
#plt.figure()
#compara2.plot(title = 'Variação comparada com a média de {} - situação em: {}'.format(selecao, ultimoDia), \
#            figsize=(20,10), kind='bar');
#plt.xlabel('Dias desde o centésimo caso')
#plt.ylabel('Variação percentual dos novos casos')
#mediaMovel = compara2.rolling(window=3).mean()[2:]
#mediaMovel.plot(title = 'Média móvel de 3 dias da variação comparada com a média de {} - situação em: {}'.format(selecao, ultimoDia), \
#            figsize=(20,10), kind='bar');
#plt.xlabel('Dias desde o centésimo caso')
#plt.ylabel('Variação percentual dos novos casos')
print("Taxas de  Crescimento")
compara2
# Constrói o gráfico comparativo da evolução nos dias a partir dos 10 óbitos

tabela2 = pd.pivot_table(novo, values='CumDeaths', index=['date', 'Country/Region'], aggfunc=np.sum)
tabela2.reset_index(inplace = True)
tabela2['Óbitos'] = tabela2.CumDeaths.astype('int32')
tabela2['Data'] = pd.to_datetime(tabela2['date'])
del tabela2['date']
del tabela2['CumDeaths']
tabela2 = tabela2[tabela2['Óbitos'] > 10]
tabela2.rename(columns = {'Country/Region' : 'País'}, inplace = True)
paises2 = tabela2['País'].unique()
tudo2 = pd.DataFrame()
for pais2 in paises2:
    pais2 = tabela2[tabela2['País'] == pais2]
    pais2.reset_index(inplace = True, drop = True)
    pais2.reset_index(inplace = True)
    tudo2 = pd.concat([tudo2, pais2], axis = 0)
tudo2.rename(columns = {'index' : 'Dias'}, inplace = True)
tudo2 = tudo2[tudo2['País'] != 'Others']
tudo2 = tudo2[tudo2['País'] != 'China']
tudo2 = tudo2[tudo2['País'].isin(selecao)]
tabela2 = pd.pivot_table(tudo2, values='Óbitos', index=['Dias'],
                    columns=['País'], aggfunc=np.sum)
tabela2 = tabela2[tabela2.index <= 10]
resumo = tabela2 
%matplotlib inline
plt.figure()
resumo.plot(style = {'Portugal' : 'k-o'}, \
            title = 'Óbitos nos 10 dias a seguir ao décimo óbito - situação em: {}'.format(ultimoDia), \
            figsize=(20,10));
plt.xlabel('Dias desde o décimo óbito')
plt.ylabel('Número de Óbitos')
plt.show();