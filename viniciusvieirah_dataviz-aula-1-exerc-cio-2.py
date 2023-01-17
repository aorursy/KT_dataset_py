import pandas as pd

from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/anv.csv', delimiter=',')
df.dtypes
df.head(10)
var_df = [

    ["aeronave_tipo_veiculo", "Qualitativa Nominal"],

    ["aeronave_fabricante", "Qualitativa Nominal"],

    ["aeronave_modelo", "Qualitativa Nominal"],

    ["aeronave_motor_tipo", "Quantitativa Nominal"],

    ["aeronave_ano_fabricacao", "Quantitativa Ordinal"],

    ["aeronave_nivel_dano", "Qualitativa Nominal"],

    ["total_fatalidades", "Quantitativa Discreta"]

]

var_df = pd.DataFrame(var_df, columns=["Variavel", "Classificação"])

var_df
for var_df, classificacao in zip(var_df['Variavel'], var_df['Classificação']):

    if "Qualitativa" in classificacao:

        display(df[var_df].value_counts().reset_index())



mortes = df['total_fatalidades'].value_counts().reset_index().head()

explode = (0.02, 0.02, 0.3,0.5,0.8)  # explode 1st slice

plt.figure(figsize=(10, 5))

plt.pie(x=mortes['total_fatalidades'],startangle=140, explode= explode, autopct='%0.1f%%', pctdistance=1.1,)

plt.title('Fatalidades')

plt.legend(labels=mortes['index'], bbox_to_anchor=(1, 0))

plt.axis('equal')

plt.show()
fabricantes = df['aeronave_fabricante'].value_counts().reset_index()

plt.figure(figsize=(100, 100))

plt.barh(y=fabricantes['index'], width=fabricantes['aeronave_fabricante'])

plt.title('Fabricantes')

plt.xlabel('Quantidade de aeronaves')

plt.ylabel('Nome Fabricante')

plt.show()

#ficou horrivel a visualizaçao, pegar somente valores maiores
fabricantes = df['aeronave_fabricante'].value_counts().reset_index().head(10)



plt.figure(figsize=(6, 6))

plt.barh(y=fabricantes['index'], width=fabricantes['aeronave_fabricante'])

plt.title('Fabricantes')

plt.xlabel('Quantidade de aeronaves')

plt.ylabel('Nome Fabricante')

plt.show()
areonave = df['aeronave_tipo_veiculo'].value_counts().reset_index().head()

explode = (0.02, 0.02, 0.1,0.2,0.5)  # explode 1st slice

plt.figure(figsize=(10, 5))

plt.pie(x=areonave['aeronave_tipo_veiculo'],startangle=140, explode = explode,autopct='%0.1f%%', pctdistance=1.1,)

plt.title('Tipo de aeronave')

plt.legend(labels=areonave['index'], bbox_to_anchor=(1, 0))

plt.axis('equal')

plt.show()
modelo = df['aeronave_modelo'].value_counts().reset_index().head(10)

plt.figure(figsize=(6, 6))

plt.barh(y=modelo['index'], width=modelo['aeronave_modelo'])

plt.title('Modelo')

plt.xlabel('Quantidade por modelo')

plt.ylabel('Modelo')

plt.show()
ano = df['aeronave_ano_fabricacao'].value_counts().reset_index()

ano['qtd'] = pd.Series([1, 2, 3, 0, -1, 4])

ano['perc'] = ano['aeronave_ano_fabricacao'] * 100 / ano['aeronave_ano_fabricacao'].sum()

ano = ano[ano['qtd'] >= 0].sort_values(by='qtd')



fig, axs = plt.subplots(ncols=2,figsize=(10, 3))

plt.suptitle('Ano de Fabricação')

plt.subplots_adjust(wspace=0, hspace=0)

rect0 = axs[0].barh(y=ano['index'], width=ano['perc'], color='orange')

axs[0].invert_xaxis()

axs[0].set_xlabel('Percentual')

rect1 = axs[1].barh(y=ano['index'], width=ano['aeronave_ano_fabricacao'])

axs[1].set_yticks([])    

axs[1].set_xlabel('Total')

plt.show()
motor = df["aeronave_motor_tipo"].value_counts().plot(kind='bar')

motor.set_title("Tipo de motor", fontsize=18)

motor.set_ylabel("quantidade", fontsize=12);
nivel_dano = df['aeronave_nivel_dano'].value_counts().reset_index()

nivel_dano['qtd'] = pd.Series([2, 4, 3, 5, 1])

nivel_dano['perc'] = nivel_dano['aeronave_nivel_dano'] * 100 / nivel_dano['aeronave_nivel_dano'].sum()

nivel_dano = nivel_dano[nivel_dano['qtd'] >= 0].sort_values(by='qtd')



fig, axs = plt.subplots(ncols=2,figsize=(10, 3))

plt.suptitle('Nível Dano')

plt.subplots_adjust(wspace=0, hspace=0)

rect0 = axs[0].barh(y=nivel_dano['index'], width=nivel_dano['perc'], color='blue')

axs[0].invert_xaxis()

axs[0].set_xlabel('Percentual')

axs[1].barh(y=nivel_dano['index'], width=nivel_dano['aeronave_nivel_dano'], color='darkblue')

axs[1].set_yticks([])    

axs[1].set_xlabel('Total')

plt.show()