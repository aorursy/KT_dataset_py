## Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

#%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

plt.rcParams['figure.figsize'] = [15, 5]

from IPython import display

from ipywidgets import interact, widgets

import seaborn as sns
file_in_kaggle='/kaggle/input/corona-virus-brazil/brazil_covid19.csv'

#file_local = './data/brazil_covid19.csv'

ufs = { 'AC': 'Acre','AL': 'Alagoas','AP': 'Amapá','AM': 'Amazonas','BA': 'Bahia','CE': 'Ceará','DF': 'Distrito Federal','ES': 'Espírito Santo','GO': 'Goiás','MA': 'Maranhão','MT': 'Mato Grosso','MS': 'Mato Grosso do Sul','MG': 'Minas Gerais','PA': 'Pará','PB': 'Paraíba','PR': 'Paraná','PE': 'Pernambuco','PI': 'Piauí','RJ': 'Rio de Janeiro','RN': 'Rio Grande do Norte','RS': 'Rio Grande do Sul','RO': 'Rondônia','RR': 'Roraima','SC': 'Santa Catarina','SP': 'São Paulo','SE': 'Sergipe','TO': 'Tocantins' }



Brazil = pd.read_csv(file_in_kaggle, sep=",", decimal=".", thousands=",")

Brazil[['hour']] = Brazil[['hour']].fillna('00:00')

Brazil['datetime'] = pd.to_datetime(Brazil['date'] + ' ' + Brazil['hour'])

del Brazil['hour']

BrazilLast = Brazil.groupby("state").tail(1)

Brazil=Brazil.merge(

    pd.DataFrame.from_dict(ufs, orient='index',columns=['state']).reset_index().rename({'index': 'uf'}, axis=1)

    , on=['state'], how='left').sort_values(by=['datetime'],ascending=False).reset_index(drop=True)





Brazil.info()
TopEstados=BrazilLast[BrazilLast['cases'] > 0].sort_values(by='cases',ascending=False).reset_index(drop=True)



fig = go.Figure(go.Bar(x=TopEstados["state"], y=TopEstados['cases'],

                      text=TopEstados['cases'],

            textposition='outside'))

fig.update_yaxes(showticklabels=False)



fig.show()
BrazilLast.groupby(['state']).sum().sort_values("cases",ascending=False).head(5).plot.bar(align='edge',fill=True)
BrazilLast[~BrazilLast['state'].isin(['São Paulo', 'Minas Gerais'])].groupby(['state']).sum().sort_values("cases",ascending=False).plot.bar()
CasosSuspeitos = BrazilLast[BrazilLast['suspects'] > 0].sort_values(by=['suspects'],ascending=False).reset_index(drop=True)

CasosConfirmados = BrazilLast[BrazilLast['cases'] > 0].sort_values(by='cases',ascending=False).reset_index(drop=True)

CasosDescartados = BrazilLast[BrazilLast['refuses'] > 0].sort_values(by='refuses',ascending=False).reset_index(drop=True)

Obitos = BrazilLast[BrazilLast['deaths'] > 0].sort_values(by='deaths',ascending=False).reset_index(drop=True)



chartcol='red'

fig = []

fig = make_subplots(rows=2, cols=2, shared_xaxes=True,

                    specs=[[{},{}],

                          [{},{}]],

                    subplot_titles=('Suspects', 'Cases', 'Refuses', 'Deaths'))







fig.add_trace(go.Bar(x=CasosConfirmados["state"], y=CasosConfirmados['suspects'], 

                     text=CasosConfirmados['suspects'], 

                     textposition='outside'), row=1,col=1)





fig.add_trace(go.Bar(x=CasosConfirmados["state"], y=CasosConfirmados['cases'], 

                     text=CasosConfirmados['cases'], 

                     textposition='outside'), row=1,col=2)



fig.add_trace(go.Bar(x=CasosDescartados["state"], y=CasosDescartados['refuses'], 

                     text=CasosDescartados['refuses'], 

                     textposition='outside'), row=2,col=1)



fig.add_trace(go.Bar(x=Obitos["state"], y=Obitos['deaths'], 

                     text=Obitos['deaths'],

                     textposition='outside'), row=2,col=2)







fig.update_layout(showlegend=False)
### colos legend

dicColors = {'suspects':'yellow','cases':'red','refuses':'green','deaths':'black'}





### methods

def PlotState(fig, state, rowN = 1): 

    CasesState = Brazil[Brazil['state'] == state]

    fig.add_trace(go.Scatter(x=CasesState["datetime"],y=CasesState['suspects'],

                             mode='lines+markers',

                             name='Suspects',

                             line=dict(color=dicColors['suspects'],width=2)),

                             row=rowN,col=1)



    fig.add_trace(go.Scatter(x=CasesState["datetime"],y=CasesState['cases'],

                             mode='lines+markers',

                             name='Cases',

                             line=dict(color=dicColors['cases'],width=2)),

                             row=rowN,col=1)



    fig.add_trace(go.Scatter(x=CasesState["datetime"],y=CasesState['refuses'],

                             mode='lines+markers',

                             name='Refuses',

                             line=dict(color=dicColors['refuses'],width=2)),

                             row=rowN,col=1)



    fig.add_trace(go.Scatter(x=CasesState["datetime"],y=CasesState['deaths'],

                             mode='lines+markers',

                             name='Deaths',

                             line=dict(color=dicColors['deaths'],width=2)),

                             row=rowN,col=1)



def PlotGraph(state):

    if state == 'Sao Paulo':

        state = 'São Paulo'

    fig = make_subplots(rows=1, cols=2,shared_xaxes=True,

                        specs=[[{"colspan": 2}, None]],

                        subplot_titles=(state,''))

    PlotState(fig, state, 1)

    fig.update_layout(height=500, showlegend=True)

    return fig
interact(PlotGraph, state = widgets.Dropdown(options=ufs, value='São Paulo', description='State'))

def PlotGraphStatic(uf):

    if uf == '':

        df = Brazil

    else:

        df = Brazil[Brazil["uf"] == uf]

    

    ax = plt.subplots(figsize=(20, 8))

    

    sns.lineplot(x='datetime', y='suspects', data=df[df['suspects'] > 0].reset_index(drop=True))

    sns.lineplot(x='datetime', y='refuses', data=df[df['refuses'] > 0].reset_index(drop=True))

    sns.lineplot(x='datetime', y='cases', data=df[df['cases'] > 0].reset_index(drop=True))

    sns.lineplot(x='datetime', y='deaths', data=df[df['deaths'] > 0].reset_index(drop=True))



    plt.legend(labels=["Supects", 'Refuses', 'Cases', 'Deaths'])

    plt.ylabel('Peoples', fontsize=12)

    plt.xlabel('Date', fontsize=12)

    plt.xticks(rotation=45, ha='right')

    plt.title(uf, fontsize=14);

    #df["state"].iloc[0]

    plt.show()
PlotGraphStatic('MS')
PlotGraphStatic('MT')
PlotGraphStatic('SP')
df1 = Brazil.sort_values(by=['state'],ascending=True).reset_index(drop=True)



dfState=pd.DataFrame(np.repeat(df1["state"].unique(), len(df1["datetime"].unique())), columns=['state'])

dfDate= pd.DataFrame(df1["datetime"].unique(),columns=['datetime']).sort_values(by=['datetime'],ascending=True).reset_index(drop=True)

dfDate["order"]=dfDate.index

dfDate=pd.concat([dfDate]*len(ufs)).reset_index(drop=True)



df2 = pd.concat([dfDate,dfState],axis=1)

df2=df2.merge(Brazil, on=['datetime','state'], how='left')

df2["state2"]=df2['state'].str.normalize('NFKD').str.encode('ascii', errors='ignore')

df2[["suspects", 'refuses', 'cases', 'deaths']]=df2[["suspects", 'refuses', 'cases', 'deaths']].fillna(0)
plt.style.use('seaborn-darkgrid')

g = sns.FacetGrid(df2, col='state2', hue='state2', col_wrap=4)

 

# Add the line over the area with the plot function

g.map(plt.plot, 'datetime', 'cases')

# Fill the area with fill_between

g.map(plt.fill_between, 'datetime', 'cases', alpha=0.2).set_titles("{col_name}").set_xticklabels(rotation=45)



# Add a title for the whole plo

plt.subplots_adjust(top=0.92)



g.fig.suptitle('Evolution of confirmed cases.')

plt.show()
import datetime

now = datetime.datetime.now()

print ("Updated: " + now.strftime("%Y-%m-%d %H:%M:%S"))