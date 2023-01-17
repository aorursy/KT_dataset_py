# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


#Leitura dos dados do arquivo



#df = pd.read_csv("/kaggle/input/base-4000-nomes-csv-com-emprego/base4milnomesprofissao.csv", sep=";") 

df = pd.read_csv("/kaggle/input/agoravaimesmo/basefinalagoravaimesmo.csv", sep=";") 
#Criamos duas faixas para idade, uma usando codigos numéricos, outra com nome 

df['Faixa_Idade'] = pd.cut(df['Idade'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=[0, 2, 3, 4,5,6,7, 8,9])

df['Faixa_Idade_name'] = pd.cut(df['Idade'], [0, 12, 18, 25, 45, 65, 80, 100], labels=['Criança', 'Adolescente', 'Jovem', 'Adulto','Meia-idade','Idoso','Ancião'])
#Imputação de dados nas colunas

df.loc[df['TinhaCartAssin'].isnull(),'TinhaCartAssin'] = 0

df.loc[df['EraServPubEst'].isnull(),'EraServPubEst'] = 0

df.loc[df['EraServPubEstAnt'].isnull(),'EraServPubEstAnt'] = 0

df.loc[df['RendMenPerc'].isnull(),'RendMenPerc'] = 0

df.loc[df['EsferaAtual'].isnull(),'EsferaAtual'] = 0

df.loc[df['EsferaAnt'].isnull(),'EsferaAnt'] = 0

df.loc[df['EstadoCivil'].isnull(),'EstadoCivil'] = 9
df
#criação de nova coluna com Mobilidade - Sim e Não
def label_mobilidade (row):

   if row['Mobilidade'] == 1 :

      return 'Sim'

   if row['Mobilidade'] == 0:

      return 'Não'

   return ''
df['Mobilidade_name'] = df.apply (lambda row: label_mobilidade(row), axis=1)
#lista de colunas para filtragem



colunas =  ['nome_ocupacao','Faixa_Idade_name','UF_name','sexo_name','setorEmpSemana_name','setorEmpAnt_name','posocu358_name','faixa_rendimento_name','estado_civil_name','esferaEmpAtual_name','esferaAnt_name','carteira_assinada_name','aposentado_name','EmpregoAtual_name', 'Mobilidade_name']
#aplicação do filtro de colunas

data = df.loc[:,colunas]
df.T

#A tabela apresenta muitos avalores nulos, principalmente nas colunas referentes à esfera de atuação no serviço público, tnato no emprego atual quanto no anterior. O campo só é preenchido quando o emprego é público.

data.info()
#Acontagem de classes para cada variavel revela que a coluna ocupação tem uma quantidade vem extesa de valores (264), seguido por UF (27). tTodos os estados estão representados. 

data.describe()
#A base filtraa  tem um total de 3997 linhas. São Paulo é o estado com mais linhas (562), seguido pelo Rio Grande do Sul (453), e por ultimo temos amapá com 5.

df['UF_name'].value_counts().reset_index()
import seaborn as sns

import matplotlib.pyplot as plt

#Impressão da tabela com a quantidade de valores para cada faixa

pd.crosstab(index=data["faixa_rendimento_name"].sort_index(), columns="Total", rownames=["Faixa salarial"],colnames=[""])     

#Obsevamos que a renda percapita dmiciliar é em grande parte dos casos está entre meio salario minimo e 2 salarios minimos

#Fora dessa faixa, ous eja, acima de 3 ou abaixo de 1/4, o percentual restante é baixo, cerca de 11,3%.
plt.figure(figsize=(10,5))

chart = sns.countplot(y="faixa_rendimento_name", hue="Mobilidade_name", data=data, order = ["Sem rendimento","Sem declaração","Até ¼ salário mínimo","Mais de ¼ até ½ salário mínimo","Mais de ½ até 1 salário mínimo","Mais de 1 até 2 salários mínimos","Mais de 2 até 3 salários mínimos","Mais de 3 até 5 salários"] )

plt.legend(loc='lower right')

plt.show()
plt.figure(figsize=(10,5))

chart = sns.countplot(y="Faixa_Idade_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()
pd.crosstab(index=data["estado_civil_name"].sort_index(), columns="Total", rownames=["Estado civil"],colnames=[""])  
plt.figure(figsize=(10,5))

chart = sns.countplot(y="estado_civil_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()
pd.crosstab(index=data["sexo_name"].sort_index(), columns="Total", rownames=["Sexo"],colnames=[""])  
plt.figure(figsize=(10,5))

chart = sns.countplot(y="sexo_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()

pd.crosstab(index=data["UF_name"].sort_index(), columns="Total", rownames=["UF"],colnames=[""])  
plt.figure(figsize=(10,15))

chart = sns.countplot(y="UF_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()
pd.crosstab(index=data["aposentado_name"].sort_index(), columns="Total", rownames=["Aposentado"],colnames=[""])  
chart = sns.countplot(y="aposentado_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()
pd.crosstab(index=data["setorEmpSemana_name"].sort_index(), columns="Total", rownames=["Setor do Emprego"],colnames=[""])  
chart = sns.countplot(y="setorEmpSemana_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()  



pd.crosstab(index=data["setorEmpAnt_name"].sort_index(), columns="Total", rownames=["Setor do Emprego Anterior"],colnames=[""])  
plt.figure(figsize=(10,5))

chart = sns.countplot(y="setorEmpAnt_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()

    

pd.crosstab(index=data["posocu358_name"].sort_index(), columns="Total", rownames=["Posição da ocupação"],colnames=[""])  
plt.figure(figsize=(10,5))

chart = sns.countplot(y="posocu358_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()

    

pd.crosstab(index=data["esferaEmpAtual_name"].sort_index(), columns="Total", rownames=["Esfera do Emprego Atual"],colnames=[""])  
plt.figure(figsize=(10,5))

chart = sns.countplot(y="esferaEmpAtual_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()

pd.crosstab(index=data["esferaAnt_name"].sort_index(), columns="Total", rownames=["Esfera do Emprego Anterior"],colnames=[""]) 
plt.figure(figsize=(10,5))

chart = sns.countplot(y="esferaAnt_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()

    

pd.crosstab(index=data["carteira_assinada_name"].sort_index(), columns="Total", rownames=["Carteira Assinada"],colnames=[""]) 
plt.figure(figsize=(10,5))

chart = sns.countplot(y="carteira_assinada_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()

    

pd.crosstab(index=data["EmpregoAtual_name"].sort_index(), columns="Total", rownames=["Emprego Atual"],colnames=[""]) 
plt.figure(figsize=(10,5))

chart = sns.countplot(y="EmpregoAtual_name", hue="Mobilidade_name", data=data)

plt.legend(loc='lower right')

plt.show()
df.info()


df.info()
#Lista de colunas para o filtro

colunas = ['UF','Sexo','Idade','Raça','EstadoCivil','EsferaAtual','EraServPubEst','SetorEmpAnt','EsferaAnt','EraServPubEstAnt','TinhaCartAssin','QtdAnosEmpAnt','PosOcu358','RendMenPerc','Aposentado','ocupacao','atividade']
#Apiicação do filtro

data2 = df.loc[:,colunas]
data2
data2['Idade'] = pd.cut(data2['Idade'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100], labels=[0, 2, 3, 4,5,6,7, 8,9])

data2.info()
#data2  = data2.drop('Idade',axis = 1)
data2
corr = data2.corr(method='spearman')

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 18))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.30, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
data2
data2.corr()
# Importing Libraries



from kmodes.kmodes import KModes
#Teste do custo para varias quantidades de clusters

# Impressão do grafico Elbow



cost = []

K = range(1,5)

for num_clusters in list(K):

    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)

    kmode.fit_predict(data2)

    cost.append(kmode.cost_)

    

plt.plot(K, cost, 'bx-')

plt.xlabel('k clusters')

plt.ylabel('Cost')

plt.title('Elbow Method For Optimal k')

plt.show()
km = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)

cluster_labels = km.fit_predict(data2)

data2['Cluster'] = cluster_labels






for col in colunas:

    plt.subplots(figsize = (15,5))

    sns.countplot(x='Cluster',hue=col, data = data2)

    plt.show()
data2['atividade'].value_counts()
#Podemos ver que no cluster 0 temos grande quantidade de trabalhadores da area de construção e infraestrutura basica. Sexo masculino

#pardos, de 1/2 ate 1 salrio meinimo per capita, no domicilio, 
fitClusters_huang