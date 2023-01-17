import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.float_format = '{:.2f}'.format



import os

print(os.listdir("../input"))
# Carregando as bases de dados

df = pd.read_csv('../input/brazilian-cities/BRAZIL_CITIES.csv', sep=';')

df_ibge = pd.read_csv('../input/mortalidade-infantil/municipios_ibge.csv')

df_mortalidade = pd.read_csv('../input/mortalidade-infantil/mortalidade_infantil.csv')
df.shape, df_ibge.shape, df_mortalidade.shape
df.info()
df_ibge.info()
df_mortalidade.info()
# Adicionando a coluna UF na tabela de municipios do IBGE

estados = {

    'AC': 'Acre',

    'AL': 'Alagoas',

    'AP': 'Amapá',

    'AM': 'Amazonas',

    'BA': 'Bahia',

    'CE': 'Ceará',

    'DF': 'Distrito Federal',

    'ES': 'Espírito Santo',

    'GO': 'Goiás',

    'MA': 'Maranhão',

    'MT': 'Mato Grosso',

    'MS': 'Mato Grosso do Sul',

    'MG': 'Minas Gerais',

    'PA': 'Pará',

    'PB': 'Paraíba',

    'PR': 'Paraná',

    'PE': 'Pernambuco',

    'PI': 'Piauí',

    'RJ': 'Rio de Janeiro',

    'RN': 'Rio Grande do Norte',

    'RS': 'Rio Grande do Sul',

    'RO': 'Rondônia',

    'RR': 'Roraima',

    'SC': 'Santa Catarina',

    'SP': 'São Paulo',

    'SE': 'Sergipe',

    'TO': 'Tocantins'

}



def recupera_sigla_uf(nome_uf):

    for sigla in estados:

        if estados[sigla] == nome_uf:

            return sigla

    return ''



df_ibge['uf'] = df_ibge['Nome_UF'].apply(recupera_sigla_uf)
# Acrescentando o campo Mortalidade Infantil na tabela de municipios do IBGE

def recuperar_mortalidade_infantil(codigo_municipio):

    aux = df_mortalidade[df_mortalidade['Código'] == codigo_municipio]

    if aux.shape[0] > 0:

        return aux.iloc[0]['Mortalidade infantil']

df_ibge['mortalidade_infantil'] = df_ibge['Código Município Completo'].apply(recuperar_mortalidade_infantil)
df_ibge.info()
# Excluindo os municipios que não possuem dados de mortalidade infantil

df_ibge = df_ibge[~df_ibge['mortalidade_infantil'].isna()]
# Acrescentando a coluna mortalidade_infantil no df

def recuperar_mortalidade_infantil(municipio, uf):

    aux = df_ibge[(df_ibge['uf'] == uf) & (df_ibge['Nome_Município'].str.upper() == municipio.upper())]

    if aux.shape[0] > 0:

        return aux.iloc[0]['mortalidade_infantil']  



df['mortalidade_infantil'] = df[['CITY', 'STATE']].apply(lambda row: recuperar_mortalidade_infantil(row['CITY'], row['STATE']), axis=1)
df.info()
# Excluindo os municipios que não possuem dados de mortalidade infantil na tabela df

df = df[~df['mortalidade_infantil'].isna()]
# Excluindo colunas que não serão utilizadas

df = df.drop(['CITY', 'LAT', 'LONG', 'ALT', 'HOTELS', 'BEDS'], axis=1)
# Transformando campos com ',' em numéricos e preenchendo os valores nulos com a média

campos_com_virgula = ['IDHM', 'IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao', 'AREA', 'GVA_AGROPEC', 'GVA_INDUSTRY', 'GVA_SERVICES', 

                      'GVA_PUBLIC', ' GVA_TOTAL ', 'TAXES', 'GDP', 'POP_GDP', 'GDP_CAPITA', 'mortalidade_infantil']



for campo in campos_com_virgula:

    df[campo] = df[campo].astype(str)

    df[campo] = df[campo].str.replace(',', '.').astype('float')

    df[campo] = df[campo].fillna(pd.notna(df[campo]).mean())
df.info()
# Transformando as colunas object em códigos de categorias

colunas_texto = ['STATE', 'REGIAO_TUR', 'CATEGORIA_TUR', 'RURAL_URBAN', 'GVA_MAIN']

for col in colunas_texto:

    df[col] = df[col].astype('category').cat.codes    
# Substituindo colunas nulas por 0

colunas_empresas = ['UBER', 'MAC', 'WAL-MART']

for col in colunas_empresas:

    df[col] = df[col].fillna(0)
# Substituindo colunas nulas pela média

for col in list(df.columns):

    df[col] = df[col].fillna(pd.notna(df[col]).mean())
df.sample(20).T
# Dividindo a base em treino e validação

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.20, random_state=42)
train.shape, test.shape
# Selecionar as colunas a serem usadas no trainamento e validação



# Lista das colunas não usadas

removed_cols = ['IDHM', 'IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao', 'IDHM Ranking 2010', 'mortalidade_infantil']

#removed_cols = ['mortalidade_infantil']



# Lista das features

feats = [c for c in df.columns if c not in removed_cols]
# Importar o modelo

from sklearn.ensemble import RandomForestRegressor
# Instanciar o modelo

rf = RandomForestRegressor(random_state=42, n_estimators=200)
# Treinar o modelo

rf.fit(train[feats], train['mortalidade_infantil'])
# Fazendo as previsões

preds = rf.predict(test[feats])
# Analisar as previsões com base na métrica



# Importando a métrica

from sklearn.metrics import mean_squared_error
# Validando as previsões

mean_squared_error(test['mortalidade_infantil'], preds)
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh(figsize = (20,40))
import seaborn as sns

sns.scatterplot(x=df['GDP_CAPITA'], y=df['mortalidade_infantil'])