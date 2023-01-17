'''Ignore deprecation and future, and user warnings.'''

import warnings as wrn

wrn.filterwarnings('ignore', category = DeprecationWarning) 

wrn.filterwarnings('ignore', category = FutureWarning) 

wrn.filterwarnings('ignore', category = UserWarning) 



'''Import basic modules.'''

import pandas as pd

import numpy as np

from scipy import stats



'''Customize visualization

Seaborn and matplotlib visualization.'''

import matplotlib.pyplot as plt

import seaborn as sns                   

sns.set_style("whitegrid") 



'''Plotly visualization .'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook



'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
'''Read in export and import data from CSV file'''

df_export = pd.read_csv('../input/india-trade-data/2018-2010_export.csv')

df_import = pd.read_csv('../input/india-trade-data/2018-2010_import.csv')

'''Export and Import data at a glance.'''

bold('**Preview of Export Data:**')

display(df_export.sample(n=5))

bold('**Preview of Import Data:**')

display(df_import.sample(n=5))
'''Variable Description'''

def description(df):

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    return summary
bold('**Variable Description of export dataset:**')

display(description(df_export))



bold('**Variable Description of import dataset:**')

display(description(df_import))
"""Let's see if export and import data contain the zero and NAN values """

bold('**Export Data with zeros:**')

display(df_export[df_export.value == 0].head(3))

bold('**Import Data with zeros:**')

display(df_import[df_import.value == 0].head(3))

bold('**Export Data with NAN:**')

display(df_export.isnull().sum())

bold('**Import Data with NAN:**')

display(df_import.isnull().sum())
'''Imputing the missing valriable'''

df_import = df_import.dropna()

df_import['country'] = df_import['country'].replace({'U S A': 'USA'})

df_import = df_import.reset_index(drop=True)



df_export = df_export.dropna()

df_export['country'] = df_export['country'].replace({'U S A': 'USA'})

df_export = df_export.reset_index(drop=True)
'''Coverting dataset in year wise'''

exp_year = df_export.groupby('year').agg({'value': 'sum'})

exp_year = exp_year.rename(columns={'value': 'Export'})

imp_year = df_import.groupby('year').agg({'value': 'sum'})

imp_year = imp_year.rename(columns={'value': 'Import'})



'''Calculating the growth of export and import'''

exp_year['Growth Rate(E)'] = exp_year.pct_change()

imp_year['Growth Rate(I)'] = imp_year.pct_change()



'''Calculating trade deficit'''

total_year = pd.concat([exp_year, imp_year], axis = 1)

total_year['Trade Deficit'] = exp_year.Export - imp_year.Import



bold('**Export/Import and Trade Balance of India**')

display(total_year)

bold('**Descriptive statistics**')

display(total_year.describe())
'''Visualization of Export and Import'''

# create trace1

trace1 = go.Bar(

                x = total_year.index,

                y = total_year.Export,

                name = "Export",

                marker = dict(color = 'rgb(55, 83, 109)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = total_year.Export)

# create trace2 

trace2 = go.Bar(

                x = total_year.index,

                y = total_year.Import,

                name = "Import",

                marker = dict(color = 'rgb(26, 118, 255)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = total_year.Import)





layout = go.Layout(hovermode= 'closest', title = 'Export/Import of Indian Trade from 2010 to 2018' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'USD (millions)'))

fig = go.Figure(data = [trace1, trace2], layout = layout)

fig.show()
'''Visualization of Export/Import Growth Rate'''

# create trace1

trace1 = go.Scatter(

                x = total_year.index,

                y = total_year['Growth Rate(E)'],

                name = "Growth Rate(E)",

                line_color='deepskyblue',

                opacity=0.8,

                text = total_year['Growth Rate(E)'])

# create trace2 

trace2 = go.Scatter(

                x = total_year.index,

                y = total_year['Growth Rate(I)'],

                name = "Growth Rate(I)",

                line_color='dimgray',

                opacity=0.8,

                text = total_year['Growth Rate(I)'])



layout = go.Layout(hovermode= 'closest', title = 'Export/Import Growth Rate of Indian Trade from 2010 to 2018' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'Growth Rate'))

fig = go.Figure(data = [trace1, trace2], layout = layout)

fig.show()
'''Visualization of Export/Import and Trade Deficit'''

trace1 = go.Bar(

                x = total_year.index,

                y = total_year.Export,

                name = "Export",

                marker = dict(color = 'rgb(55, 83, 109)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = total_year.Export)

# create trace2 

trace2 = go.Bar(

                x = total_year.index,

                y = total_year.Import,

                name = "Import",

                marker = dict(color = 'rgb(26, 118, 255)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = total_year.Import)

# create trace3

trace3 = go.Bar(

                x = total_year.index,

                y = total_year['Trade Deficit'],

                name = "Trade Deficit",

                marker = dict(color = 'crimson',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = total_year['Trade Deficit'])



layout = go.Layout(hovermode= 'closest', title = 'Export/Import and Trade Deficit of Indian Trade from 2010 to 2018' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'USD (millions)'))

fig = go.Figure(data = [trace1, trace2, trace3], layout = layout)

fig.show()
'''Commodity export/Import count'''

print('Total number of Export commodity:', df_export['Commodity'].nunique())

print('Total number of Import commodity:', df_import['Commodity'].nunique())
"""Let's count the most importing and exporting commodities"""

bold('**Most Exporting Commodities(In Numbers) from 2010 to 2018**')

display(pd.DataFrame(df_export['Commodity'].value_counts().head(20)))

bold('**Most Importing Commodities(In Numbers) from 2010 to 2018**')

display(pd.DataFrame(df_import['Commodity'].value_counts().head(20)))
'''Coverting dataset in commodity wise'''

exp_comm = df_export.groupby('Commodity').agg({'value':'sum'})

exp_comm = exp_comm.sort_values(by = 'value', ascending = False)

exp_comm = exp_comm[:20]



imp_comm = df_import.groupby('Commodity').agg({'value':'sum'})

imp_comm = imp_comm.sort_values(by = 'value', ascending = False)

imp_comm = imp_comm[:20]
'''Visualization of Export/Import Commodity wise'''

def bar_plot(x,y, xlabel, ylabel, label, color):

    global ax

    font_size = 30

    title_size = 60

    plt.rcParams['figure.figsize'] = (40, 30)

    ax = sns.barplot(x, y, palette = color)

    ax.set_xlabel(xlabel = xlabel, fontsize = font_size)

    ax.set_ylabel(ylabel = ylabel, fontsize = font_size)

    ax.set_title(label = label, fontsize = title_size)

    plt.xticks(fontsize=30)

    plt.yticks(fontsize=30)

    plt.show()

    

bar_plot(exp_comm.value, exp_comm.index, 'USD (millions)', 'Commodities', 'Export of India (Commodity wise from 2010 to 2018)', 'gist_rainbow')

bar_plot(imp_comm.value, exp_comm.index, 'USD (millions)', 'Commodities', 'Import of India (Commodity wise from 2010 to 2018)', 'rainbow')
'''Create pivot table of export/import (commodity wise)'''

exp_comm_table = pd.pivot_table(df_export, values = 'value', index = 'Commodity', columns = 'year')

imp_comm_table = pd.pivot_table(df_import, values = 'value', index = 'Commodity', columns = 'year')

bold('**Commodity Composition of Exports**')

display(exp_comm_table.sample(n=5))

bold('**Commodity Composition of Imports**')

display(imp_comm_table.sample(n=5))
bold('**Trend of the Most Exporting Goods(In Values) From 2010 to 2018**')

plt.figure(figsize=(15,19))

 

plt.subplot(411)

g = exp_comm_table.loc["MINERAL FUELS, MINERAL OILS AND PRODUCTS OF THEIR DISTILLATION; BITUMINOUS SUBSTANCES; MINERAL WAXES."].plot(color='green', linewidth=3)

g.set_ylabel('USD (millions)', fontsize = 15)

g.set_xlabel('Year', fontsize = 15)

g.set_title('Trend of Petroleum products', size = 20)



plt.subplot(412)

g1 = exp_comm_table.loc["NATURAL OR CULTURED PEARLS,PRECIOUS OR SEMIPRECIOUS STONES,PRE.METALS,CLAD WITH PRE.METAL AND ARTCLS THEREOF;IMIT.JEWLRY;COIN."].plot(color='green', linewidth=3)

g1.set_ylabel('USD (millions)', fontsize = 15)

g1.set_xlabel('Year', fontsize = 15)

g1.set_title('Trend of Gems & Jewellery', size = 20)



plt.subplot(413)

g2 = exp_comm_table.loc["VEHICLES OTHER THAN RAILWAY OR TRAMWAY ROLLING STOCK, AND PARTS AND ACCESSORIES THEREOF."].plot(color='green', linewidth=3)

g2.set_ylabel('USD (millions)', fontsize = 15)

g2.set_xlabel('Year', fontsize = 15)

g2.set_title('Trend of Transport Equipment', size = 20)





plt.subplot(414)

g3 = exp_comm_table.loc["NUCLEAR REACTORS, BOILERS, MACHINERY AND MECHANICAL APPLIANCES; PARTS THEREOF."].plot(color='green', linewidth=3)

g3.set_ylabel('USD (millions)', fontsize = 15)

g3.set_xlabel('Year', fontsize = 15)

g3.set_title('Trend of Machinery & Nuclear Reactors', size = 20)



plt.subplots_adjust(hspace = 0.4)

plt.show()
bold('**Trend of the Most Importing Goods(In Values) From 2010 to 2018**')

plt.figure(figsize=(15,19))

 

plt.subplot(411)

g = imp_comm_table.loc["MINERAL FUELS, MINERAL OILS AND PRODUCTS OF THEIR DISTILLATION; BITUMINOUS SUBSTANCES; MINERAL WAXES."].plot(color='red', linewidth=3)

g.set_ylabel('USD (millions)', fontsize = 15)

g.set_xlabel('Year', fontsize = 15)

g.set_title('Trend of Petroleum products', size = 20)



plt.subplot(412)

g1 = imp_comm_table.loc["NATURAL OR CULTURED PEARLS,PRECIOUS OR SEMIPRECIOUS STONES,PRE.METALS,CLAD WITH PRE.METAL AND ARTCLS THEREOF;IMIT.JEWLRY;COIN."].plot(color='red', linewidth=3)

g1.set_ylabel('USD (millions)', fontsize = 15)

g1.set_xlabel('Year', fontsize = 15)

g1.set_title('Trend of Gems & Jewellery', size = 20)



plt.subplot(413)

g2 = imp_comm_table.loc["ELECTRICAL MACHINERY AND EQUIPMENT AND PARTS THEREOF; SOUND RECORDERS AND REPRODUCERS, TELEVISION IMAGE AND SOUND RECORDERS AND REPRODUCERS,AND PARTS."].plot(color='red', linewidth=3)

g2.set_ylabel('USD (millions)', fontsize = 15)

g2.set_xlabel('Year', fontsize = 15)

g2.set_title('Trend of Electrical Equipment', size = 20)





plt.subplot(414)

g3 = imp_comm_table.loc["NUCLEAR REACTORS, BOILERS, MACHINERY AND MECHANICAL APPLIANCES; PARTS THEREOF."].plot(color='red', linewidth=3)

g3.set_ylabel('USD (millions)', fontsize = 15)

g3.set_xlabel('Year', fontsize = 15)

g3.set_title('Trend of Machinery & Nuclear Reactors', size = 20)



plt.subplots_adjust(hspace = 0.4)

plt.show()
'''Country export/Import count'''

print('Total number of country Export to:', df_export['country'].nunique())

print('Total number of country Import from:', df_import['country'].nunique())
'''Coverting dataset in Country wise'''

exp_country = df_export.groupby('country').agg({'value':'sum'})

exp_country = exp_country.rename(columns={'value': 'Export'})

exp_country = exp_country.sort_values(by = 'Export', ascending = False)

exp_country = exp_country[:20]



imp_country = df_import.groupby('country').agg({'value':'sum'})

imp_country = imp_country.rename(columns={'value': 'Import'})

imp_country = imp_country.sort_values(by = 'Import', ascending = False)

imp_country = imp_country[:20]
'''Visualization of Export/Import Country wise'''

bar_plot(exp_country.Export, exp_country.index, 'USD (millions)', 'Countries', 'Export of India (Country wise from 2010 to 2018)', 'plasma')

bar_plot(imp_country.Import, imp_country.index, 'USD (millions)', 'Countries', 'Import of India (Country wise from 2010 to 2018)', 'viridis')
'''Calculating trade deficit'''

total_country = pd.concat([exp_country, imp_country], axis = 1)

total_country['Trade Deficit'] = exp_country.Export - imp_country.Import

total_country = total_country.sort_values(by = 'Trade Deficit', ascending = False)

total_country = total_country[:11]



bold('**Direction of Foreign Trade Export/Import and Trade Balance of India**')

display(total_country)

bold('**Descriptive statistics**')

display(total_country.describe())
'''Visualization of Export/Import and Trade Deficit'''

trace1 = go.Bar(

                x = total_country.index,

                y = total_country.Export,

                name = "Export",

                marker = dict(color = 'rgb(55, 83, 109)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = total_year.Export)

# create trace2 

trace2 = go.Bar(

                x = total_country.index,

                y = total_country.Import,

                name = "Import",

                marker = dict(color = 'rgb(26, 118, 255)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = total_year.Import)

# create trace3

trace3 = go.Bar(

                x = total_country.index,

                y = total_country['Trade Deficit'],

                name = "Trade Deficit",

                marker = dict(color = 'crimson',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = total_year['Trade Deficit'])



layout = go.Layout(hovermode= 'closest', title = 'Export/Import and Trade Deficit of Indian Trade from 2010 to 2018(Country Wise)' , xaxis = dict(title = 'Country'), yaxis = dict(title = 'USD (millions)'))

fig = go.Figure(data = [trace1, trace2, trace3], layout = layout)

fig.show()
'''Create pivot table of export/import (country wise)'''

exp_country_table = pd.pivot_table(df_export, values = 'value', index = 'country', columns = 'year')

imp_country_table = pd.pivot_table(df_import, values = 'value', index = 'country', columns = 'year')

bold('**Direction of Foreign Trade Export in India**')

display(exp_country_table.sample(n=5))

bold('**Direction of Foreign Trade Import in India**')

display(imp_country_table.sample(n=5))
bold('**Trend of the Direction of Foreign Trade Export in India From 2010 to 2018**')

plt.figure(figsize=(15,19))

 

plt.subplot(411)

g = exp_country_table.loc["USA"].plot(color='purple', linewidth=3)

g.set_ylabel('USD (millions)', fontsize = 15)

g.set_xlabel('Year', fontsize = 15)

g.set_title('Trend of Export to the USA', size = 20)



plt.subplot(412)

g1 = exp_country_table.loc["U ARAB EMTS"].plot(color='purple', linewidth=3)

g1.set_ylabel('USD (millions)', fontsize = 15)

g1.set_xlabel('Year', fontsize = 15)

g1.set_title('Trend of Export to the UAE', size = 20)



plt.subplot(413)

g2 = exp_country_table.loc["CHINA P RP"].plot(color='purple', linewidth=3)

g2.set_ylabel('USD (millions)', fontsize = 15)

g2.set_xlabel('Year', fontsize = 15)

g2.set_title('Trend of Export to the China', size = 20)





plt.subplot(414)

g3 = exp_country_table.loc["HONG KONG"].plot(color='purple', linewidth=3)

g3.set_ylabel('USD (millions)', fontsize = 15)

g3.set_xlabel('Year', fontsize = 15)

g3.set_title('Trend of Export to the Hong Kong', size = 20)



plt.subplots_adjust(hspace = 0.4)

plt.show()
bold('**Trend of the Direction of Foreign Trade Import in India From 2010 to 2018**')

plt.figure(figsize=(15,19))

 

plt.subplot(411)

g = imp_country_table.loc["USA"].plot(color='coral', linewidth=3)

g.set_ylabel('USD (millions)', fontsize = 15)

g.set_xlabel('Year', fontsize = 15)

g.set_title('Trend of Import From the USA', size = 20)



plt.subplot(412)

g1 = imp_country_table.loc["U ARAB EMTS"].plot(color='coral', linewidth=3)

g1.set_ylabel('USD (millions)', fontsize = 15)

g1.set_xlabel('Year', fontsize = 15)

g1.set_title('Trend of Import From the UAE', size = 20)



plt.subplot(413)

g2 = imp_country_table.loc["CHINA P RP"].plot(color='coral', linewidth=3)

g2.set_ylabel('USD (millions)', fontsize = 15)

g2.set_xlabel('Year', fontsize = 15)

g2.set_title('Trend of Import From the China', size = 20)





plt.subplot(414)

g3 = imp_country_table.loc["SAUDI ARAB"].plot(color='coral', linewidth=3)

g3.set_ylabel('USD (millions)', fontsize = 15)

g3.set_xlabel('Year', fontsize = 15)

g3.set_title('Trend of Import From the Saudi Arab', size = 20)



plt.subplots_adjust(hspace = 0.4)

plt.show()
''' creating a new dataframe on Sections of HSCode'''

HSCode = pd.DataFrame() 

HSCode['Start']=[1,6,15,16,25,28,39,41,44,47,50,64,68,71,72,84,86,90,93,94,97]

HSCode['End']=[5,14,15,24,27,38,40,43,46,49,63,67,70,71,83,85,89,92,93,96,99]

HSCode['Sections Name']=['Animals & Animal Products',

'Vegetable Products',

'Animal Or Vegetable Fats',

'Prepared Foodstuffs',

'Mineral Products',

'Chemical Products',

'Plastics & Rubber',

'Hides & Skins',

'Wood & Wood Products',

'Wood Pulp Products',

'Textiles & Textile Articles',

'Footwear, Headgear',

'Articles Of Stone, Plaster, Cement, Asbestos',

'Pearls, Precious Or Semi-Precious Stones, Metals',

'Base Metals & Articles Thereof',

'Machinery & Mechanical Appliances',

'Transportation Equipment',

'Instruments - Measuring, Musical',

'Arms & Ammunition',

'Miscellaneous',

'Works Of Art',]

HSCode.index += 1

HSCode.index.name = 'Section'
bold('**List Of indian HS Classification is based on the HS Code:**')

display(HSCode)
df_export['Sections Name'] = df_export['HSCode']

df_import['Sections Name'] = df_import['HSCode']

for i in range(1,22):

    df_export.loc[(df_export['Sections Name'] >= HSCode['Start'][i]) & (df_export['Sections Name'] <= HSCode['End'][i]),'Sections Name']=i

    df_import.loc[(df_import['Sections Name'] >= HSCode['Start'][i]) & (df_import['Sections Name'] <= HSCode['End'][i]),'Sections Name']=i
exp_hscode = df_export.groupby(['Sections Name']).agg({'value':'sum'})

exp_hscode['Sections Name'] = HSCode['Sections Name']

imp_hscode = df_import.groupby(['Sections Name']).agg({'value':'sum'})

imp_hscode['Sections Name'] = HSCode['Sections Name']
'''Visualization of Export/Import HS Classification is based'''

def bar_plot(x,y, xlabel, ylabel, label, color):

    global ax

    font_size = 30

    title_size = 60

    plt.rcParams['figure.figsize'] = (40, 30)

    ax = sns.barplot(x, y, palette = color)

    ax.set_xlabel(xlabel = xlabel, fontsize = font_size)

    ax.set_ylabel(ylabel = ylabel, fontsize = font_size)

    ax.set_title(label = label, fontsize = title_size)

    plt.xticks(rotation = 90, fontsize=30)

    plt.yticks(fontsize=30)

    plt.show()



bar_plot(exp_hscode['Sections Name'], exp_hscode.value, 'Sections Name', 'USD (millions)', 'Export of India (HS Classification is based)', 'Paired')

bar_plot(imp_hscode['Sections Name'], imp_hscode.value, 'Sections Name', 'USD (millions)', 'Import of India (HS Classification is based)', 'Set1')
plt.rcParams['figure.figsize'] = (20, 7)

plt.style.use('seaborn-dark-palette')



sns.boxenplot(df_export['Sections Name'], df_export['value'], palette = 'coolwarm')

plt.title('Comparison of Exports and HS Classification', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (20, 7)

plt.style.use('seaborn-dark-palette')



sns.boxenplot(df_import['Sections Name'], df_import['value'], palette = 'gist_heat_r')

plt.title('Comparison of Import and HS Classification', fontsize = 20)

plt.show()
'Top expensive import and export '

expensive_import = df_import.sort_values(by='value',  ascending=False).head(100)



import squarify

import matplotlib

temp1 = expensive_import.groupby(['country']).agg({'value': 'sum'})

temp1 = temp1.sort_values(by='value')



norm = matplotlib.colors.Normalize(vmin=min(expensive_import.value), vmax=max(expensive_import.value))

colors = [matplotlib.cm.Blues(norm(value)) for value in expensive_import.value]



value=np.array(temp1)

country=temp1.index

fig = plt.gcf()

ax = fig.add_subplot()

fig.set_size_inches(16, 4.5)

squarify.plot(sizes=value, label=country, color = colors, alpha=.6)

plt.title("Expensive Imports Countrywise Share", fontweight="bold")

plt.axis('off')

plt.show()
expensive_import = df_import.sort_values(by='value',  ascending=False).head(500)

temp2 = expensive_import.groupby(['HSCode']).agg({'value': 'sum'})

temp2 = temp2.sort_values(by='value')



norm = matplotlib.colors.Normalize(vmin=min(expensive_import.value), vmax=max(expensive_import.value))

colors = [matplotlib.cm.plasma(norm(value)) for value in expensive_import.value]



value=np.array(temp2)

country=temp2.index

fig = plt.gcf()

ax = fig.add_subplot()

fig.set_size_inches(16, 4.5)

squarify.plot(sizes=value, label=country, color = colors, alpha=.6)

plt.title("Expensive Imports Commodity (HSCode)", fontweight="bold")

plt.axis('off')

plt.show()
import warnings

warnings.filterwarnings('ignore')



export_map = pd.DataFrame(df_export.groupby(['country'])['value'].sum().reset_index())

count = pd.DataFrame(export_map.groupby('country')['value'].sum().reset_index())



trace = [go.Choropleth(

            colorscale = 'algae',

            locationmode = 'country names',

            locations = count['country'],

            text = count['country'],

            z = count['value'],

            reversescale=True)]



layout = go.Layout(title = 'India Export to Other Country')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)
import_map = pd.DataFrame(df_import.groupby(['country'])['value'].sum().reset_index())

count = pd.DataFrame(import_map.groupby('country')['value'].sum().reset_index())



trace = [go.Choropleth(

            colorscale = 'amp',

            locationmode = 'country names',

            locations = count['country'],

            text = count['country'],

            z = count['value'],

            reversescale=True)]



layout = go.Layout(title = 'India Import from Other Country')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)
df_final_trade = pd.concat([df_export, df_import])

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (12, 12)

wordcloud = WordCloud(background_color = 'gray', width = 1200,  height = 1200, max_words = 100).generate(' '.join(df_final_trade['country']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Country in Trade With India',fontsize = 30)

plt.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (12, 12)

wordcloud = WordCloud(background_color = 'tan', width = 1200,  height = 1200, max_words = 100).generate(' '.join(df_final_trade['Commodity']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Commodity in Trade With India',fontsize = 30)

plt.show()