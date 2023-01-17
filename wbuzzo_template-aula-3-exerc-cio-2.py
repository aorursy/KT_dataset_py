import plotly
import plotly.graph_objs as go
import pandas as pd

#Lendo o dataset que será utilizado - BR_eleitorado_2016_municipio
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

#Agrupador - Qtde total de eleitores por estado
eleitores_uf = (df.groupby(['uf']).sum()["total_eleitores"]).reset_index()

#Gerando o grafico de linhas
plotly.offline.init_notebook_mode(connected=True)

random_x = eleitores_uf['uf']
random_y = eleitores_uf['total_eleitores'].round()

trace1 = go.Scatter(
    x = random_x,
    y = random_y,
)

data = [trace1]
plotly.offline.iplot(data, filename='basic-line')
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import pandas as pd

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

#Lendo o dataset que será utilizado - BR_eleitorado_2016_municipio
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

#Agrupador - Qtde total de eleitores por genero
eleitores_masc = (df.groupby(['uf']).sum()["gen_masculino"]).reset_index()
eleitores_fem = (df.groupby(['uf']).sum()["gen_feminino"]).reset_index()

trace1 = go.Bar(
    x=eleitores_uf['uf'],
    y=eleitores_masc["gen_masculino"],
    #text=eleitores_masc["gen_masculino"],
    #textposition = 'auto',
    marker=dict(
        color='rgb(0,176,240)'
    ),
    name='Masc'
)

trace2 = go.Bar(
    x=eleitores_uf['uf'],
    y=eleitores_fem["gen_feminino"],
    #text=eleitores_fem["gen_feminino"],
    #textposition = 'auto',
    marker=dict(
        color='rgb(255,0,0)'
    ),
    name='Fem'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='grouped-bar')
import plotly.plotly as py
import plotly.graph_objs as go

#Lendo o dataset que será utilizado - BR_eleitorado_2016_municipio
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

#Agrupador - uf
eleitores_uf = (df.groupby(['uf']).sum()["total_eleitores"]).reset_index()

#Gerando histograma
x = eleitores_uf["total_eleitores"]
data = [go.Histogram(x=x)]

plotly.offline.iplot(data, filename='basic histogram')
import pandas as pd
import numpy as np

#Lendo o dataset que será utilizado - BR_eleitorado_2016_municipio
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

#Criando o DataFrame com dados para classificação por Região
regiao = [["Norte","AM"],["Norte","RR"],["Norte","AP"],["Norte","PA"],["Norte","TO"],["Norte","RO"],["Norte","AC"],["Nordeste","MA"],
["Nordeste","PI"],["Nordeste","CE"],["Nordeste","RN"],["Nordeste","PE"],["Nordeste","PB"],["Nordeste","SE"],["Nordeste","AL"],
["Nordeste","BA"],["Centro-Oeste","MT"],["Centro-Oeste","MS"],["Centro-Oeste","GO"],["Sudeste","SP"],["Sudeste","RJ"],["Sudeste","ES"],
["Sudeste","MG"],["Sul","PR"],["Sul","RS"],["Sul","SC"]]
df_reg = pd.DataFrame(regiao, columns=["regiao", "uf"])

#Criando o Dataset
x = pd.merge(df, df_reg, on='uf')

#Retirando os campos que não serão utilizados
x.drop('cod_municipio_tse', axis=1, inplace=True)
x.drop('nome_municipio', axis=1, inplace=True)
x.drop('total_eleitores', axis=1, inplace=True)
x.drop('gen_masculino', axis=1, inplace=True)
x.drop('gen_feminino', axis=1, inplace=True)
x.drop('gen_nao_informado', axis=1, inplace=True)

#Transformando as colunas do dataset para formatação da analise
x = pd.melt(x, id_vars=['uf', 'regiao'], var_name='faixa_etaria', value_name='valor')
x = pd.DataFrame(x[['uf','regiao','faixa_etaria','valor']].groupby(by=['uf','regiao','faixa_etaria'], as_index=False).sum())

x.head(10)
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import pandas as pd

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

#Delimitando o dataframe
y = pd.DataFrame(x[['regiao','valor']].groupby(by=['regiao'], as_index=False).sum())

#Criando o grafico de radar por região
data = [go.Scatterpolar(
  r = y["valor"],
  theta = y["regiao"],
  fill = 'toself'
)]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True      
    )
  ),
  showlegend = False
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename = "radar/basic")
import plotly.plotly as py
import plotly.graph_objs as go

#Delimitando o dataframe
y = pd.DataFrame(x[['regiao','uf','valor']].groupby(by=['regiao','uf'], as_index=False).sum())

#Craindo o boxplot
trace1 = go.Box(y = y[y["regiao"]=="Norte"]["valor"], name = "Norte")
trace2 = go.Box(y = y[y["regiao"]=="Nordeste"]["valor"], name = "Nordeste")
trace3 = go.Box(y = y[y["regiao"]=="Centro-Oeste"]["valor"], name = "Centro-Oeste")
trace4 = go.Box(y = y[y["regiao"]=="Sudeste"]["valor"], name = "Sudeste")
trace5 = go.Box(y = y[y["regiao"]=="Sul"]["valor"], name = "Sul")

data = [trace1, trace2, trace3, trace4, trace5]
plotly.offline.iplot(data)
import pandas as pd
import numpy as np

#Lendo o dataset que será utilizado - BR_eleitorado_2016_municipio
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

#Criando o DataFrame com dados para classificação por Região
regiao = [["Norte","AM"],["Norte","RR"],["Norte","AP"],["Norte","PA"],["Norte","TO"],["Norte","RO"],["Norte","AC"],["Nordeste","MA"],
["Nordeste","PI"],["Nordeste","CE"],["Nordeste","RN"],["Nordeste","PE"],["Nordeste","PB"],["Nordeste","SE"],["Nordeste","AL"],
["Nordeste","BA"],["Centro-Oeste","MT"],["Centro-Oeste","MS"],["Centro-Oeste","GO"],["Sudeste","SP"],["Sudeste","RJ"],["Sudeste","ES"],
["Sudeste","MG"],["Sul","PR"],["Sul","RS"],["Sul","SC"]]
df_reg = pd.DataFrame(regiao, columns=["regiao", "uf"])

#Criando o Dataset
x = pd.merge(df, df_reg, on='uf')

#Retirando os campos que não serão utilizados
x.drop('cod_municipio_tse', axis=1, inplace=True)
x.drop('nome_municipio', axis=1, inplace=True)
x.drop('total_eleitores', axis=1, inplace=True)
x.drop('gen_nao_informado', axis=1, inplace=True)

#Transformando as colunas do dataset para formatação da analise
x = pd.melt(x, id_vars=['uf','regiao','gen_masculino','gen_feminino'], var_name='faixa_etaria', value_name='valor')

x.drop('faixa_etaria', axis=1, inplace=True)
x.drop('valor', axis=1, inplace=True)

x = pd.melt(x, id_vars=['regiao','uf'], var_name='genero', value_name='valor')
x = pd.DataFrame(x[['regiao','uf','genero','valor']].groupby(by=['regiao','uf','genero'], as_index=False).sum())

x.head(10)
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd

df = x

#Criando o grafico de violino
fig = {
    "data": [
        {
            "type": 'violin',
            "x": df['regiao'][df['genero'] == 'gen_masculino'],
            "y": df['valor'][df['genero'] == 'gen_masculino'],
            "legendgroup": 'Masculino',
            "scalegroup": 'Masculino',
            "name": 'Masculino',
            "side": 'negative',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'blue'
            }
        },
        {
            "type": 'violin',
            "x": df['regiao'][df['genero'] == 'gen_feminino'],
            "y": df['valor'][df['genero'] == 'gen_feminino'],
            "legendgroup": 'Feminino',
            "scalegroup": 'Feminino',
            "name": 'Feminino',
            "side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'red'
            }
        }
    ],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violingap": 0,
        "violinmode": "overlay"
    }
}


plotly.offline.iplot(fig, filename = 'violin/split', validate = False)