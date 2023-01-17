## Um gráfico de linha
### Usando o dataset de cars e as colunas de Classification e Years
import pandas as pd
url ="https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/cars.csv"
car = pd.read_csv(url, sep=',')
car.head()
ClassificationYear = car[['Classification', 'Year']].groupby(['Year'])['Classification'].value_counts().rename_axis(['Year','Classification2']).reset_index()
ClassificationYear.columns = ['Year', 'Classification', 'total']
ClassificationYear

import plotly as p
import plotly.plotly as py
import plotly.graph_objs as go
p.tools.set_credentials_file(username='vfousas', api_key='xZdH3n5eIaZDB8eJ4iTs')
trace0 = go.Scatter(
      x = ClassificationYear[ClassificationYear['Classification']=='Automatic transmission']['Year'],
      y = ClassificationYear[ClassificationYear['Classification']=='Automatic transmission']['total'],
      name ='Automatic transmission',
      line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4))

trace1 = go.Scatter(
      x = ClassificationYear[ClassificationYear['Classification']=='Manual transmission']['Year'],
      y = ClassificationYear[ClassificationYear['Classification']=='Manual transmission']['total'],
      name ='Manual transmission',
      line = dict(
        color = ('rgb(0, 255, 100)'),
        width = 4))
data = [trace0, trace1]

layout = dict(title = 'Comparação entre vendas de carros com transmissão manual e automática ao longo dos anos',
              xaxis = dict(title = 'Years'),
              yaxis = dict(title = 'Total Sales'),
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-line')

FuelTypeYear = car[['Fuel Type', 'Year']].groupby(['Year'])['Fuel Type'].value_counts().rename_axis(['Year','Classification2']).reset_index()
FuelTypeYear.columns = ['Year', 'Fuel Type', 'total']
FuelTypeYear
trace0 = go.Bar(
      x = FuelTypeYear[FuelTypeYear['Fuel Type']=='Gasoline']['Year'],
      y = FuelTypeYear[FuelTypeYear['Fuel Type']=='Gasoline']['total'],
      name ='Gasoline')

trace1 = go.Bar(
     x = FuelTypeYear[FuelTypeYear['Fuel Type']=='Diesel fuel']['Year'],
     y = FuelTypeYear[FuelTypeYear['Fuel Type']=='Diesel fuel']['total'],
     name ='Diesel fuel')

    
trace2 = go.Bar(
      x = FuelTypeYear[FuelTypeYear['Fuel Type']=='E85']['Year'],
      y = FuelTypeYear[FuelTypeYear['Fuel Type']=='E85']['total'],
      name ='E85')

trace3 = go.Bar(
      x = FuelTypeYear[FuelTypeYear['Fuel Type']=='Compressed natural gas']['Year'],
      y = FuelTypeYear[FuelTypeYear['Fuel Type']=='Compressed natural gas']['total'],
      name ='Compressed natural gas	')

data = [trace0, trace1, trace2, trace3]

layout = dict(title = 'Comparação entre vendas de carros por tipo de combustivel',
              xaxis = dict(title = 'Years'),
              yaxis = dict(title = 'Total Sales'),
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar-direct-labels')
url ="https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/milsa.csv"
milsa = pd.read_csv(url, sep=',')
milsa.head()
trace0 = go.Histogram(
      x = milsa['Salario'],
      name ='Salario')

data = [trace0]
layout = dict(title = 'Histograma de salários',
              yaxis = dict(title = 'Qnt de salario'))
              
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='stacked histogram')
url ="https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/cota_parlamentar_sp.csv"
cota = pd.read_csv(url, sep=';')
cota.head()
partidoPT = cota[cota['sgpartido']=='PT']
partidoPT = partidoPT[partidoPT['numano']==2015].groupby(['nummes']).sum()
partidoPT = partidoPT['vlrdocumento'].reset_index()
partidoPT
data = [
    go.Scatterpolar(
      r = partidoPT['vlrdocumento'],
      theta = ["Jan", "Fev", "Abr","Mar", "Mai", "Jun","Jul", "Ago", "Set","Out", "Nov", "Dez"],
      fill = "toself")
    ]
layout = go.Layout(
 
  polar = dict(
    radialaxis = dict(
      visible = True
    )
  ),
  title='Total de gastos por mês do ano 2015 do Partido PT',
  showlegend = True
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename = "polar'")
partidoPT = cota[cota['sgpartido']=='PT']
partidoPT = partidoPT.groupby(['nummes']).sum()
partidoPT = partidoPT['vlrdocumento'].reset_index()
partidoPT
data = [
    go.Box(
      y = partidoPT['vlrdocumento'])
    ]
layout = go.Layout(
  title='Total geral de gastos do Partido PT',
  showlegend = False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename = "polar'")
url ="https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/formatted_flights.csv"
flights = pd.read_csv(url, sep=',')
flights.head()
United  = flights[flights['name']=='United Air Lines Inc.']['arr_delay'].reset_index().drop('index', axis=1)
United
fig = {
    "data": [{
        "type": 'violin',
        "y": United['arr_delay'],
        "box": {
            "visible": True
        },
        "line": {
            "color": 'black'
        },
        "meanline": {
            "visible": True
        },
        "fillcolor": '#8dd3c7',
        "opacity": 0.6,
        "x0": 'Total Delay'
    }],
    "layout" : {
        "title": "Atraso na aterrisagem de voos da empres United Airlines",
        "yaxis": {
            "zeroline": False,
        }
    }
}

py.iplot(fig, filename = 'violin', validate = False)