# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
despesas = pd.read_sas('/kaggle/input/eleies/eleicao_2018_candidato_despesa.sas7bdat',format = 'sas7bdat', encoding='ISO-8859-1')

receitas = pd.read_sas('/kaggle/input/eleies/eleicao_2018_candidato_receita.sas7bdat',format = 'sas7bdat', encoding='ISO-8859-1')

final = pd.read_sas('/kaggle/input/eleies/eleicao_2018_apuracao_final.sas7bdat',format = 'sas7bdat', encoding='ISO-8859-1')
pd.options.display.float_format = '{:20,.2f}'.format

#despesas.head()
despesas_final.info()
# Join com despesa final

despesas_final = pd.merge(despesas, final, on='SQ_CANDIDATO')

despesas_final_1_turno = despesas_final[despesas_final['NR_TURNO'] == 1]

despesas_final_2_turno = despesas_final[despesas_final['NR_TURNO'] == 2]
despesas.describe().T
# Quantitativo de despesa por origem

despesas.groupby('DS_ORIGEM_DESPESA')['VR_DESPESA_CONTRATADA'].sum().reset_index().sort_values('VR_DESPESA_CONTRATADA',ascending = False )
# Quantitativo de despesa por candidato

despesas_final.groupby(['SQ_CANDIDATO', 'NM_CANDIDATO', 'DS_CARGO'])['VR_DESPESA_CONTRATADA'].sum().reset_index().sort_values('VR_DESPESA_CONTRATADA',ascending = False )
#-- Aumentando área do gráfico

plt.figure(figsize=(15,5))



#-- Colocando título

plt.title('Valor da despesa por origem')

#-- Rotação de 90º nos labels

plt.xticks(rotation=90)

sns.barplot(data = despesas, x='DS_TIPO_DOCUMENTO', y='VR_DESPESA_CONTRATADA')



plt.show()
#filtrar primeiro turno e sq candidato

contadorCargos = despesas_final_1_turno['DS_CARGO'].value_counts().reset_index()

contadorCargos
# create data

plt.figure(figsize=(7,7))

plt.title('Distribuição de Candidatos por Cargo')







names=contadorCargos['index']

size_of_groups=contadorCargos['DS_CARGO'] # Filtrar por turno e sq







# Create a pieplot

plt.pie(size_of_groups )

#plt.show()

plt.legend(names,loc=3)

# add a circle at the center

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

 

plt.show()





plt.figure(figsize=(18,5))

plt.xticks(rotation=90)

plt.title('Distribuição de despesas ao longo do tempo')

sns.lineplot(x="DT_DESPESA", y="Despesa_Total", data=despesas_final)
# Quantidade de votos por raça

votos_por_cor = despesas_final_1_turno.groupby(['DS_COR_RACA'])['Votos'].sum().reset_index().sort_values('Votos',ascending = False )
votos_por_cor # ajustar formatacao
#-- Aumentando área do gráfico

plt.figure(figsize=(15,5))



#-- Colocando título

plt.title('Quantidade de votos por Cor_Raça')

#-- Rotação de 90º nos labels

#plt.xticks(rotation=90)

sns.barplot(data = votos_por_cor, x='DS_COR_RACA', y='Votos')



plt.show()
plt.figure(figsize=(8,8))

plt.title('Correlação entre as variáveis')

sns.heatmap(despesas_final.corr(), linewidths=.5, cmap="YlGnBu")
uf_por_receita = despesas_final_1_turno.groupby(['SG_UF'])['Receita_Total'].sum().reset_index().sort_values('Receita_Total',ascending = False )
uf_por_receita # separar por turno
sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(6, 15))



# Load the example car crash dataset

#crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)



plt.title('Receita Total por Estado')



# Plot the crashes where alcohol was involved

sns.set_color_codes("muted")

sns.barplot(x="Receita_Total", y="SG_UF", data=uf_por_receita,

            label="Alcohol-involved", color="green")
uf_por_votos = despesas_final_1_turno.groupby(['SG_UF'])['Votos'].sum().reset_index().sort_values('Votos',ascending = False )
uf_por_votos
sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(6, 15))



# Load the example car crash dataset

#crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)



plt.title('Quantidade de Votos por Estado')



# Plot the crashes where alcohol was involved

sns.set_color_codes("muted")

sns.barplot(x="Votos", y="SG_UF", data=uf_por_votos,

            label="Votos", color="b")
candidatos_uf = despesas_final_1_turno.groupby(['SG_UF'])['SQ_CANDIDATO'].count().reset_index().sort_values('SQ_CANDIDATO',ascending = False )
candidatos_uf
sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(6, 15))



# Load the example car crash dataset

#crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)



plt.title('Quantidade de candidatos por Estado')



# Plot the crashes where alcohol was involved

sns.set_color_codes("muted")

sns.barplot(x="SQ_CANDIDATO", y="SG_UF", data=candidatos_uf,

            label="Quantidade Candidatos", color="c")