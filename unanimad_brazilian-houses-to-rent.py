import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots
df = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

figs = []
df.head()
df.info()
df.describe()
temp = df[['city', 'animal', 'area']].groupby(['city', 'animal'], as_index=False).count().sort_values(by=['area'], ascending=False)



fig = go.Figure()



animals = ['acept', 'not acept']



for x in animals:

    fig.add_trace(go.Bar(

        x=temp['city'].loc[temp['animal'] == x], y=temp['area'].loc[temp['animal'] == x], 

        text=temp['area'].loc[temp['animal'] == x], textposition='auto', name=x,

    ))



fig.update_layout(title='Quantidade de casas por cidade separadas por permitir ou não animal de extimação', barmode='stack')

figs.append(fig)
temp = df[['city', 'furniture', 'area']].groupby(['city', 'furniture'], as_index=False).count().sort_values(by=['area'], ascending=False)



fig = go.Figure()



furniture = ['furnished', 'not furnished']



for x in furniture:

    fig.add_trace(go.Bar(

        x=temp['city'].loc[temp['furniture'] == x], y=temp['area'].loc[temp['furniture'] == x], 

        text=temp['area'].loc[temp['furniture'] == x], textposition='auto', name=x,

    ))



fig.update_layout(title='Quantidade de casas mobiliadas por cidade', barmode='stack')

figs.append(fig)
temp = df[['city', 'area', 'total (R$)']].groupby('city', as_index=False).mean().sort_values(by='area', ascending=False)



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=temp['city'], values=temp['area']), row=1, col=1)

fig.add_trace(go.Pie(labels=temp['city'], values=temp['total (R$)']), row=1, col=2)



fig.update_traces(textinfo='value')

fig.update_layout(title='Média de área e valor do aluguel por cidade')



figs.append(fig)
temp = df[['city', 'hoa (R$)', 'rent amount (R$)', 'property tax (R$)', 'fire insurance (R$)', 'total (R$)']].groupby('city', as_index=False).mean().sort_values(by='total (R$)', ascending=False)



taxes = ['hoa (R$)', 'rent amount (R$)', 'property tax (R$)', 'fire insurance (R$)']



fig = go.Figure()



for tax in taxes:

    fig.add_trace(go.Bar(x=temp['city'], y=temp[tax], name=tax))

    

fig.update_layout(title='Distribuição das taxas dos alugueis por cidade', barmode='stack', )

figs.append(fig)
df[['area', 'rooms', 'bathroom', 'parking spaces']]
fig = go.Figure()



fig.add_trace(go.Heatmap(x=['rooms', 'bathroom'],

    z=df[['rooms', 'bathroom']].groupby('rooms', as_index=False).mean().sort_values(by='rooms')))



fig.update_layout(title='Relação da quantidade de quartos e banheiros dos imóveis')



figs.append(fig)
fig = go.Figure()



fig.add_trace(go.Box(y=df['area']))



figs.append(fig)
fig = go.Figure()



fig.add_trace(go.Scatter(x=df['area'].loc[df['animal'] == 'acept'], y=df['total (R$)'].loc[df['animal'] == 'acept'], mode='markers', name='Animal Acept'))

fig.add_trace(go.Scatter(x=df['area'].loc[df['animal'] == 'not acept'], y=df['total (R$)'].loc[df['animal'] == 'not acept'], mode='markers', name='Animal Not Acept'))



fig.update_layout(title='Distribuição dos imóveis por área e valor de aluguel que aceitam ou não animais de estimação', xaxis_title='Area', yaxis_title='Valor (R$)')



figs.append(fig)
figs[0].show()
figs[1].show()
figs[2].show()
figs[3].show()
figs[4].show()
figs[5].show()
figs[6].show()