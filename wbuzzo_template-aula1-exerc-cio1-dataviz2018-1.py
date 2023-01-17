import pandas as pd
import numpy as np

#Lendo o Dataset
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

#Criando o DataFrame com dados para classificação por Região (analise incremental)
regiao = [["Norte","AM"],["Norte","RR"],["Norte","AP"],["Norte","PA"],["Norte","TO"],["Norte","RO"],["Norte","AC"],["Nordeste","MA"],
["Nordeste","PI"],["Nordeste","CE"],["Nordeste","RN"],["Nordeste","PE"],["Nordeste","PB"],["Nordeste","SE"],["Nordeste","AL"],
["Nordeste","BA"],["Centro-Oeste","MT"],["Centro-Oeste","MS"],["Centro-Oeste","GO"],["Sudeste","SP"],["Sudeste","RJ"],["Sudeste","ES"],
["Sudeste","MG"],["Sul","PR"],["Sul","RS"],["Sul","SC"]]
df_reg = pd.DataFrame(regiao, columns=["regiao", "uf"])

#Criando o Dataset
df = pd.merge(df, df_reg, on='uf')

#Retirando os campos que não serão utilizados
df.drop('cod_municipio_tse', axis=1, inplace=True)
df.drop('total_eleitores', axis=1, inplace=True)
df.drop('nome_municipio', axis=1, inplace=True)

#Transformando as colunas do dataset para formatação da analise
df = pd.melt(df, id_vars=['uf', 'regiao', 'gen_masculino', 'gen_feminino', 'gen_nao_informado'], var_name='faixa_etaria', value_name='valor')
df = pd.melt(df, id_vars=['uf', 'regiao', 'faixa_etaria','valor'], var_name='genero', value_name='valor_gen')

#Criando os dataframes
df = pd.DataFrame(df[['uf','regiao','faixa_etaria','valor','genero','valor_gen']].groupby(by=['uf','regiao','faixa_etaria','genero','valor_gen'], as_index=False).sum())
df = pd.DataFrame(df[['uf','regiao','faixa_etaria','valor','genero','valor_gen']].groupby(by=['uf','regiao','faixa_etaria','valor','genero'], as_index=False).sum())

#Classificando as variaveis
resposta = [["uf", "Categorica Nominal"],["regiao", "Categorica Nominal"],["faixa_etaria", "Categorica Ordinal"],["valor", "Numerica Discreta"],["genero", "Categorica Nominal"],["valor_gen", "Numerica Discreta"]]

#Criando o dataframe de classificação
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
#Regiao | Frequencia absoluta
reg_qtd = [df['regiao'].value_counts()]
reg_qtd = pd.DataFrame(reg_qtd).unstack().reset_index(name='Freq')
reg_qtd.drop('level_1', axis=1, inplace=True)
reg_qtd.columns = ['Regiao', 'Freq']

#Regiao | Frequencia relativa
reg_per = [round(df['regiao'].value_counts()/df.shape[0]*100,2)]
reg_per = pd.DataFrame(reg_per).unstack().reset_index(name='Freq')
reg_per.drop('level_1', axis=1, inplace=True)
reg_per.columns = ['Regiao', '%']

pd.merge(reg_qtd, reg_per, on='Regiao')
#Estado | Frequencia absoluta
uf_qtd = [df['uf'].value_counts()]
uf_qtd = pd.DataFrame(uf_qtd).unstack().reset_index(name='Freq')
uf_qtd.drop('level_1', axis=1, inplace=True)
uf_qtd.columns = ['uf', 'Freq']

#Estado | Frequencia relativa
uf_per = [round(df['uf'].value_counts()/df.shape[0]*100,2)]
uf_per = pd.DataFrame(uf_per).unstack().reset_index(name='Freq')
uf_per.drop('level_1', axis=1, inplace=True)
uf_per.columns = ['uf', '%']

pd.merge(uf_qtd, uf_per, on='uf')
#Faixa Etaria | Frequencia absoluta
fx_qtd = [df['faixa_etaria'].value_counts()]
fx_qtd = pd.DataFrame(fx_qtd).unstack().reset_index(name='Freq')
fx_qtd.drop('level_1', axis=1, inplace=True)
fx_qtd.columns = ['fx', 'Freq']

#Faixa Etaria | Frequencia relativa
fx_per = [round(df['faixa_etaria'].value_counts()/df.shape[0]*100,2)]
fx_per = pd.DataFrame(fx_per).unstack().reset_index(name='Freq')
fx_per.drop('level_1', axis=1, inplace=True)
fx_per.columns = ['fx', '%']

pd.merge(fx_qtd, fx_per, on='fx')
#Genero | Frequencia absoluta
gen_qtd = [df['genero'].value_counts()]
gen_qtd = pd.DataFrame(gen_qtd).unstack().reset_index(name='Freq')
gen_qtd.drop('level_1', axis=1, inplace=True)
gen_qtd.columns = ['gen', 'Freq']

#Genero | Frequencia relativa
gen_per = [round(df['genero'].value_counts()/df.shape[0]*100,2)]
gen_per = pd.DataFrame(gen_per).unstack().reset_index(name='Freq')
gen_per.drop('level_1', axis=1, inplace=True)
gen_per.columns = ['gen', '%']

pd.merge(gen_qtd, gen_per, on='gen')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure
#Definição do Grid para plotagem dos graficos
gridsize = (3, 3) #3 rows, 3 columns
fig = plt.figure(figsize=(12, 12))

gen = plt.subplot2grid(gridsize, (0, 0), colspan=1)
reg = plt.subplot2grid(gridsize, (0, 1), colspan=2)
uf = plt.subplot2grid(gridsize, (1, 0), colspan=3)
faixa = plt.subplot2grid(gridsize, (2, 0), colspan=3)


#Comparativo eleitores por Genero
x, y = sorted(df['genero'].unique(), key=None), df.groupby(by=['genero'])['valor_gen'].sum()/(df['valor_gen'].sum())*100
gen.set_title('(%) Eleitores por Genero')
gen.pie(y, labels = round(y,1), shadow = True)
#gen.legend(x)
gen.legend(['Fem','Masc', 'NI'], loc="best")


#Comparativo eleitores por Regiao
x, y = sorted(df['regiao'].unique(), key=None), df.groupby(by=['regiao'])['valor_gen'].sum()
reg.set_title('Eleitores por Região')
reg.bar(x, y, color = 'b')


#Comparativo eleitores por Estado
x, y = sorted(df['uf'].unique(), key=None), df.groupby(by=['uf'])['valor_gen'].sum()
uf.set_title('Eleitores por Estado')
uf.bar(x, y, color = 'b')                                  


#Comparativo eleitores por Faixa Etaria
x, y = sorted(df['faixa_etaria'].unique(), key=None), df.groupby(by=['faixa_etaria'])['valor'].sum()
faixa.set_title('Eleitores por Faixa Etaria')
faixa.bar(x, y, color = 'b')