import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()
%matplotlib inline

# import data
se = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')
shsat = pd.read_csv('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')
evict = pd.read_csv('../input/2017-2018-evictions/Evictions.csv')
finance = pd.read_csv('../input/nyc-financial-empowerment-centers/financial-empowerment-centers.csv')
emerg = pd.read_csv('../input/ny-emergency-response-incidents/emergency-response-incidents.csv')

# clean up school income estimate
se['School Income Estimate'] = se['School Income Estimate'].astype(str).str.replace('$','')
se['School Income Estimate'] = se['School Income Estimate'].astype(str).str.replace(',','')
se['School Income Estimate'] = se['School Income Estimate'].astype(str).str.replace(' ','')
se['School Income Estimate'] = se['School Income Estimate'].apply(lambda x: float(x))

# dropping null values
nullIncome = se[se['School Income Estimate'].isnull()]
df1 = se.drop(nullIncome.index)
nullNeed = df1[df1['Economic Need Index'].isnull()]
df1 = df1.drop(nullNeed.index)

# create borough series
df1['Borough'] = df1['City']
for i in df1[df1['City'] == 'NEW YORK'].index:
    df1.at[i, 'Borough'] = 'MANHATTAN'
for i in df1[df1['City'] == 'ROOSEVELT ISLAND'].index:
    df1.at[i, 'Borough'] = 'MANHATTAN'
for i in df1[df1['City'] == 'ELMHURST'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'WOODSIDE'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'CORONA'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'MIDDLE VILLAGE'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'MASPETH'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'RIDGEWOOD'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'GLENDALE'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'LONG ISLAND CITY'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'FLUSHING'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'COLLEGE POINT'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'WHITESTONE'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'BAYSIDE'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'QUEENS VILLAGE'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'LITTLE NECK'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'DOUGLASTON'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'FLORAL PARK'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'BELLEROSE'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'JAMAICA'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'ARVERNE'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'FAR ROCKAWAY'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'SOUTH OZONE PARK'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'BROAD CHANNEL'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'RICHMOND HILL'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'WOODHAVEN'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'SOUTH RICHMOND HILL'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'OZONE PARK'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'ROCKAWAY PARK'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'HOWARD BEACH'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'ROCKAWAY BEACH'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'KEW GARDENS'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'FOREST HILLS'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'REGO PARK'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'SPRINGFIELD GARDENS'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'HOLLIS'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'SAINT ALBANS'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'ROSEDALE'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'CAMBRIA HEIGHTS'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'JACKSON HEIGHTS'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'ASTORIA'].index:
    df1.at[i, 'Borough'] = 'QUEENS'
for i in df1[df1['City'] == 'EAST ELMHURST'].index:
    df1.at[i, 'Borough'] = 'QUEENS'

# convert percent to float
def toFloat(x):
    fin = int(x.replace('%', ''))
    return fin
df1['Percent Asian'] = df1['Percent Asian'].astype(str).apply(toFloat)
df1['Percent Black'] = df1['Percent Black'].astype(str).apply(toFloat)
df1['Percent Hispanic'] = df1['Percent Hispanic'].astype(str).apply(toFloat)
df1['Percent White'] = df1['Percent White'].astype(str).apply(toFloat)
df1['Percent ELL'] = df1['Percent ELL'].astype(str).apply(toFloat)
df1['Percent Black / Hispanic'] = df1['Percent Black / Hispanic'].astype(str).apply(toFloat)

# df1[['Percent Asian', 'Percent Black', 'Percent Hispanic', 'Percent White']].head()
# (nullIncome[nullIncome['School Income Estimate'].isnull()].shape[0] + nullNeed[nullNeed['Economic Need Index'].isnull()].shape[0])/se.shape[0]

# convert float/int to string for labeling
df1['incomeString'] = df1['School Income Estimate'].apply(lambda x: str(x))
df1['econString'] = df1['Economic Need Index'].apply(lambda x: str(round(x,2)))
df1['blackString'] = df1['Percent Black'].apply(lambda x: str(x))
df1['hispanicString'] = df1['Percent Hispanic'].apply(lambda x: str(x))
df1['asianString'] = df1['Percent Asian'].apply(lambda x: str(x))
df1['whiteString'] = df1['Percent White'].apply(lambda x: str(x))
df1['ellString'] = df1['Percent ELL'].apply(lambda x: str(x))
df1['blackhispanicString'] = df1['Percent Black / Hispanic'].apply(lambda x: str(x))
df1['Percent White / Asian'] = df1['Percent Asian'] + df1['Percent White']

# mean DF
cityNumbers = pd.DataFrame()
cityNumbers['Count'] = df1.groupby('Borough')['Percent Black'].count().sort_values(ascending=False)
cityNumbers['ENI Mean'] = df1.groupby('Borough')['Economic Need Index'].mean()
cityNumbers['percentBlackMean'] = df1.groupby('Borough')['Percent Black'].mean()
cityNumbers['percentHispanicMean'] = df1.groupby('Borough')['Percent Hispanic'].mean()

# cleaning up eviction df
evict['EXECUTED_DATE'] = pd.to_datetime(evict['EXECUTED_DATE'])
evict.drop(evict[evict['EXECUTED_DATE'] == evict['EXECUTED_DATE'].max()].index, inplace=True)
evictRes = evict[evict['RESIDENTIAL_COMMERCIAL_IND'] == 'Residential']
evicted = evictRes.groupby('BOROUGH')['COURT_INDEX_NUMBER'].count().values


# financial empowerment centers
finProv = finance.groupby('Borough')['Provider'].count().values

# def cleanUpERI():
# nyc emergency response incidents
emerg['Creation Date'] = pd.to_datetime(emerg['Creation Date'])
emerg['Closed Date'] = pd.to_datetime(emerg['Closed Date'])
emerg['Incident Category'] = emerg['Incident Type'].apply(lambda x: x.split('-')[0])
# clean up borough names
for i in emerg[emerg['Borough'] == 'BRonx'].index:
     emerg.at[i, 'Borough'] = 'bronx'
for i in emerg[emerg['Borough'] == 'BrONX'].index:
     emerg.at[i, 'Borough'] = 'bronx'
for i in emerg[emerg['Borough'] == 'Bronx (NYCHA)'].index:
     emerg.at[i, 'Borough'] = 'bronx'
for i in emerg[emerg['Borough'] == 'Bronx'].index:
     emerg.at[i, 'Borough'] = 'bronx'
for i in emerg[emerg['Borough'] == 'Brooklyn (NYCHA-Brevoort)'].index:
     emerg.at[i, 'Borough'] = 'brooklyn'
for i in emerg[emerg['Borough'] == 'Brooklyn'].index:
     emerg.at[i, 'Borough'] = 'brooklyn'
for i in emerg[emerg['Borough'] == 'Essex'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Mamhattan'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Manahttan'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Manahttan'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Manhatan'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Manhattah'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'MANHATTAN'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Manhattan (Pier 92)'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Manhattan (Waldorf Astoria)'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Manhatten'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Manhhattan'].index:
     emerg.at[i, 'Borough'] = 'manhattan'     
for i in emerg[emerg['Borough'] == 'Manhaatan'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Manhattan'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Manhttan'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Mnahattan'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Nassau'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'New York'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'new york'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'New York/Manhattan'].index:
     emerg.at[i, 'Borough'] = 'manhattan'  
for i in emerg[emerg['Borough'] == 'New Yotk'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'NewYork'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'nyc'].index:
     emerg.at[i, 'Borough'] = 'manhattan'
for i in emerg[emerg['Borough'] == 'Astoria'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'quenns'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'astoria'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'Far Rockaway'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'QUEENS'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'Jamaice'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'Queens'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'Flushing'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'Hollis'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'Jamaica'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'Long Island City'].index:
     emerg.at[i, 'Borough'] = 'queens'
for i in emerg[emerg['Borough'] == 'staten Island'].index:
     emerg.at[i, 'Borough'] = 'staten island'
for i in emerg[emerg['Borough'] == 'Richmond/Staten Island'].index:
     emerg.at[i, 'Borough'] = 'staten island'
for i in emerg[emerg['Borough'] == 'SI'].index:
     emerg.at[i, 'Borough'] = 'staten island'
for i in emerg[emerg['Borough'] == 'Richmond/Staten Island'].index:
     emerg.at[i, 'Borough'] = 'Staten Island (Midland Beach Area)'
for i in emerg[emerg['Borough'] == 'Staten ISland'].index:
     emerg.at[i, 'Borough'] = 'staten island'
for i in emerg[emerg['Borough'] == 'Staten Island'].index:
     emerg.at[i, 'Borough'] = 'Staten Island (Midland Beach Area)'
for i in emerg[emerg['Borough'] == 'Staten Island (Midland Beach Area)'].index:
     emerg.at[i, 'Borough'] = 'staten island'
for i in emerg[emerg['Borough'] == 'Staten island'].index:
     emerg.at[i, 'Borough'] = 'staten island'
emerg.drop(emerg[emerg['Borough'] == 'Bergen'].index, inplace=True)
emerg.drop(emerg[emerg['Borough'] == 'Hoboken'].index, inplace=True)
emerg.drop(emerg[emerg['Borough'] == 'Citywide'].index, inplace=True)
for i in emerg[emerg['Incident Category'] == 'LawEnforcement'].index:
     emerg.at[i, 'Incident Category'] = 'Law Enforcement'


# import os
# os.listdir('../input/ny-emergency-response-incidents/emergency-response-incidents.csv')
# Looking at the data, there are null values for School Income Estimate and the Economic Need Index which account for 32.63% of the data which leaves us with 857 useable records
data = [
    {
        'x': df1['Longitude'],
        'y': df1['Latitude'],
        'mode': 'markers',
        'text': df1['School Name'] + ', ' + df1['Borough'],
        'marker': {
            'size': df1['School Income Estimate']/4500,
            'color': df1['Economic Need Index'],
            'showscale': True,
            'colorscale': 'Portland',
            'colorbar': {
                'title': 'ENI',
                'ticks': 'outside'
            }
        }
    }
]

layout= go.Layout(
    title= 'New York School Income with Economic Need',
    xaxis= {
        'title' : 'Longitude'
    },
    yaxis={
        'title': 'Latitude'
    }
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
# Lowest school income
df1.loc[df1['School Income Estimate'].idxmin()]
# highest economic need
df1.loc[df1['Economic Need Index'].idxmax()]
data = [
    {
        'x': df1['Percent White / Asian'],
        'y': df1['Economic Need Index'],
        'mode': 'markers',
        'text': df1['blackhispanicString'] + '% Black/Hispanic, ' + df1['ellString'] + '% ELL',
        'marker': {
            'color': df1['School Income Estimate'],
            'showscale': True,
            'colorscale': 'Portland',
            'colorbar': {
                'title': 'School Income',
                'ticks': 'outside'
            }
        }
    }
]

layout= go.Layout(
    title= 'Percentage of White/Asian Students Vs Economic Need Index',
    xaxis= {
        'title' : 'Percentage of White/Asian Students'
    },
    yaxis={
        'title': 'Economic Need Index'
    }
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = [
    {
        'x': df1['Percent Black / Hispanic'],
        'y': df1['Economic Need Index'],
        'mode': 'markers',
        'text': df1['blackhispanicString'] + '% Black/Hispanic, ' + df1['ellString'] + '% ELL',
        'marker': {
            'color': df1['School Income Estimate'],
            'showscale': True,
            'colorscale': 'Portland',
            'colorbar': {
                'title': 'School Income',
                'ticks': 'outside'
            }
        }
    }
]

layout= go.Layout(
    title= 'Percentage of Black/Hispanic Students Vs Economic Need Index',
    xaxis= {
        'title' : 'Percentage of Black/Hispanic Students'
    },
    yaxis={
        'title': 'Economic Need Index'
    }
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


data = [
    {
        'x': df1['Longitude'],
        'y': df1['Latitude'],
        'mode': 'markers',
        'text': df1['Borough'] + ', Economic Need: ' + df1['econString'] + ', Percent Black: ' + df1['blackString'],
        'marker': {
            'color': df1['Economic Need Index'],
            'size': df1['Percent Black']/3,
            'showscale': True,
            'colorscale': 'Portland',
            'colorbar': {
                'title': 'Economic Need',
                'ticks': 'outside'
            }
        }
    }
]

layout= go.Layout(
    title= '% Black and Economic Need',
    xaxis= {
        'title' : 'Longitude'
    },
    yaxis={
        'title': 'Latitude'
    }
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = [
    {
        'x': df1['Longitude'],
        'y': df1['Latitude'],
        'mode': 'markers',
        'text': df1['Borough'] + ', Economic Need: ' + df1['econString'] + ', Percent Hispanic: ' + df1['hispanicString'],
        'marker': {
            'color': df1['Economic Need Index'],
            'size': df1['Percent Hispanic']/3,
            'showscale': True,
            'colorscale': 'Portland',
            'colorbar': {
                'title': 'Economic Need',
                'ticks': 'outside'
            }
        }
    }
]

layout= go.Layout(
    title= '% Hispanic and Economic Need',
    xaxis= {
        'title' : 'Longitude'
    },
    yaxis={
        'title': 'Latitude'
    }
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

trace1 = go.Bar(
    x= ['Brookyln', 'Queens', 'Bronx','Manhattan', 'Staten Island'],
    y= [cityNumbers['ENI Mean'][0]*100, cityNumbers['ENI Mean'][1]*100, cityNumbers['ENI Mean'][2]*100, cityNumbers['ENI Mean'][3]*100, cityNumbers['ENI Mean'][4]*100],
    name = '% ENI Mean')

trace2 = go.Bar(
    x= ['Brookyln', 'Queens', 'Bronx','Manhattan', 'Staten Island'],
    y= [cityNumbers['percentBlackMean'][0], cityNumbers['percentBlackMean'][1], cityNumbers['percentBlackMean'][2], cityNumbers['percentBlackMean'][3], cityNumbers['percentBlackMean'][4]],
    name = '% Black Mean')

trace3 = go.Bar(
    x= ['Brookyln', 'Queens', 'Bronx','Manhattan', 'Staten Island'],
    y= [cityNumbers['percentHispanicMean'][0], cityNumbers['percentHispanicMean'][1], cityNumbers['percentHispanicMean'][2], cityNumbers['percentHispanicMean'][3], cityNumbers['percentHispanicMean'][4]],
    name = '% Hispanic Mean')



data = [trace1,trace2,trace3]

layout = go.Layout(
    barmode='group',
    title= 'Mean of ENI, Black Students, Hispanic Students by Borough',
    xaxis= {'title': 'Areas'},
    yaxis= {'title': 'Percent'}
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

trace1 = go.Bar(
    x= ['Bronx','Brookyln','Manhattan', 'Staten Island', 'Queens'],
    y= [evicted[0], evicted[1], evicted[2], evicted[4], evicted[3]],
    name = '% ENI Mean')

data = [trace1]

layout = go.Layout(
    title= 'Number of Residential Evictions by Borough 1/2017- 7/2018',
    xaxis= {'title': 'Boroughs'},
    yaxis= {'title': 'Number of Evictions'}
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
bor = ['Bronx','Brookyln','Manhattan', 'Staten Island', 'Queens']
nycPop = ['1,471,160', '2,648,771', '1,664,727', '479,458', '2,358,582']

pd.DataFrame(index=bor, data={'Population':nycPop})
trace1 = go.Bar(
    x= ['Bronx','Brookyln','Manhattan', 'Staten Island', 'Queens'],
    y= [1471160/evicted[0], 2648771/evicted[1], 1664727/evicted[2], 479458/evicted[4], 2358582/evicted[3]],
    text= ['Population: 1,471,160', 'Population: 2,648,771', 'Population: 1,664,727', 'Population: 479,458', 'Population: 2,358,582', ],
    name = '% ENI Mean')

data = [trace1]

layout = go.Layout(
    title= 'Number of People per Residential Eviction by Borough 1/2017- 7/2018',
    xaxis= {'title': 'Boroughs'},
    yaxis= {'title': 'Number of People per Eviction'}
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
finance.groupby('Borough')['Provider'].count()
trace1 = go.Bar(
    x= ['Bronx','Brookyln','Manhattan', 'Staten Island', 'Queens'],
    y= [42.47/finProv[0], 69.5/finProv[1], 22.82/finProv[2], 58.69/finProv[4], 108.1/finProv[3]],
    textposition = 'auto',
    name = 'mi^2 per Center')

trace2 = go.Bar(
    x= ['Bronx','Brookyln','Manhattan', 'Staten Island', 'Queens'],
    y= [147.1160/finProv[0], 264.8771/finProv[1], 166.4727/finProv[2], 47.9458/finProv[4], 235.8582/finProv[3]],
    textposition = 'auto',
    name = 'people(10k)/Center')

trace3 = go.Bar(
    x= ['Bronx','Brookyln','Manhattan', 'Staten Island', 'Queens'],
    y= [cityNumbers['ENI Mean'][2]*100, cityNumbers['ENI Mean'][0]*100, cityNumbers['ENI Mean'][3]*100, cityNumbers['ENI Mean'][4]*100, cityNumbers['ENI Mean'][1]*100],
    textposition = 'auto',
    name = 'Avg ENI')

data = [trace1,trace2,trace3]

layout = go.Layout(
    barmode= 'group',
    title= 'Financial Empowerment Centers',
    xaxis= {'title': 'Boroughs'}
    # ,yaxis= {'title': 'Number of People per Eviction'}
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

bor = emerg.groupby(['Incident Category','Borough']).count().index.levels[1].tolist()
mlGroup = emerg.groupby(['Incident Category','Borough']).count()['Incident Type']
inCatList = emerg.groupby(['Incident Category','Borough']).count().index.levels[0].tolist()

trace0 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[0]][0], mlGroup.loc[inCatList[0]][1],mlGroup.loc[inCatList[0]][2], mlGroup.loc[inCatList[0]][3]],
    name = inCatList[0])
trace1 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[1]][0], mlGroup.loc[inCatList[1]][1]],
    name = inCatList[1])
trace2 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[2]][0], mlGroup.loc[inCatList[2]][1],mlGroup.loc[inCatList[2]][2], mlGroup.loc[inCatList[2]][3], mlGroup.loc[inCatList[2]][4]],
    name = inCatList[2])
trace3 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[3]][0], mlGroup.loc[inCatList[3]][1],mlGroup.loc[inCatList[3]][2], mlGroup.loc[inCatList[3]][3], mlGroup.loc[inCatList[3]][4]],
    name = inCatList[3])
trace4 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[4]][0], mlGroup.loc[inCatList[4]][1],mlGroup.loc[inCatList[4]][2], mlGroup.loc[inCatList[4]][3], mlGroup.loc[inCatList[4]][4]],
    name = inCatList[4])
trace5 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[5]][0], mlGroup.loc[inCatList[5]][1],mlGroup.loc[inCatList[5]][2], mlGroup.loc[inCatList[5]][3], mlGroup.loc[inCatList[5]][4]],
    name = inCatList[5])
trace6 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[6]][0], mlGroup.loc[inCatList[6]][1],mlGroup.loc[inCatList[6]][2], mlGroup.loc[inCatList[6]][3],mlGroup.loc[inCatList[6]][4]],
    name = inCatList[6])
trace7 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[7]][0], mlGroup.loc[inCatList[7]][1],mlGroup.loc[inCatList[7]][2], mlGroup.loc[inCatList[7]][3], mlGroup.loc[inCatList[7]][4]],
    name = inCatList[7])
trace8 = go.Bar(
    x= [bor[4]],
    y= [mlGroup.loc[inCatList[8]][0]],
    name = inCatList[8])
trace9 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[9]][0], mlGroup.loc[inCatList[9]][1],mlGroup.loc[inCatList[9]][2], mlGroup.loc[inCatList[9]][3], mlGroup.loc[inCatList[9]][4]],
    name = inCatList[9])
trace10 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[10]][0], mlGroup.loc[inCatList[10]][1],mlGroup.loc[inCatList[10]][2], mlGroup.loc[inCatList[10]][3], mlGroup.loc[inCatList[10]][4]],
    name = inCatList[10])
trace11 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[11]][0], mlGroup.loc[inCatList[11]][1],mlGroup.loc[inCatList[11]][2], mlGroup.loc[inCatList[11]][3], mlGroup.loc[inCatList[11]][4]],
    name = inCatList[11])
trace12 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[12]][0], mlGroup.loc[inCatList[12]][1],mlGroup.loc[inCatList[12]][2], mlGroup.loc[inCatList[12]][3], mlGroup.loc[inCatList[12]][4]],
    name = inCatList[12])
trace13 = go.Bar(
    x= [bor[0],bor[1],bor[2],bor[3],bor[4]],
    y= [mlGroup.loc[inCatList[13]][0], mlGroup.loc[inCatList[13]][1],mlGroup.loc[inCatList[13]][2], mlGroup.loc[inCatList[13]][3], mlGroup.loc[inCatList[13]][4]],
    name = inCatList[13])

data = [trace0,trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11,trace12,trace13]

layout = go.Layout(
    barmode='stack',
    title= 'Emergency Response Incident Types by Borough',
    xaxis= {'title': 'Borough'},
    yaxis= {'title': 'Number of Incidents'}
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = [
    {
        'x': df1['Percent Black / Hispanic'],
        'y': df1['Percent ELL'],
        'mode': 'markers',
        'text': df1['blackhispanicString'] + '% Black/Hispanic, ' + df1['ellString'] + '% ELL',
        'marker': {
            'colorscale': 'Portland'
            }
        }
]

layout= go.Layout(
    title= 'Percentage of Black/Hispanic Students Vs Percentage of ELL students',
    xaxis= {
        'title' : 'Percentage of Black/Hispanic Students'
    },
    yaxis={
        'title': 'Percentage of ELL students'
    }
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
print('Number of schools with over 80% Black/Hispanic\n\tStudents and have ELL students: ' + str(df1[(df1['Percent Black / Hispanic'] > 80) & (df1['Percent ELL'] >= 20)].shape[0]))
print('Total number of schools: ' + str(df1.shape[0]))
print('Percentage of schools with over 80% Black/Hispanic\n\tStudents and have ELL students: ' + str(round((df1[(df1['Percent Black / Hispanic'] > 80) & (df1['Percent ELL'] >= 20)].shape[0]/df1.shape[0])*100, 2)) + '%')
ellBH = df1.copy()
ellBH.drop(ellBH[(ellBH['Percent Black / Hispanic'] < 80) | (ellBH['Percent ELL'] < 20)].index, inplace=True)

data = [
    {
        'x': ellBH['Longitude'],
        'y': ellBH['Latitude'],
        'mode': 'markers',
        'text': ellBH['Borough'] + ', ' + ellBH['ellString'] + '% ELL' + ', ' + ellBH['blackhispanicString'] + '% Black/Hispanic',
        'marker': {
            'size': ellBH['Percent Black / Hispanic']/4,
            'color': ellBH['Percent ELL'],
            'showscale': True,
            'colorscale': 'Portland',
            'colorbar': {
                'title': '% ELL',
                'ticks': 'outside'
            }
        }
    }
]

layout= go.Layout(
    title= 'Schools with >80% Black/Hispanic Students Vs Percentage of ELL students',
    xaxis= {
        'title' : 'Longitude'
    },
    yaxis={
        'title': 'Latitude'
    }
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
df1[(df1['Percent Black / Hispanic'] > 80) & (df1['Percent ELL'] > 30) & (df1['Borough'] == 'BRONX')][['Borough','School Name', 'Economic Need Index', 'Percent ELL', 'Percent Black', 'Percent Hispanic']].sort_values('Economic Need Index', ascending=False)