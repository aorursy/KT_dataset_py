import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import rc
import seaborn as sns
from scipy.optimize import curve_fit
import altair as alt
df=pd.read_excel('/kaggle/input/produccion-energetica-noruega/Datos Noruega.xlsx')
display(df.describe())
df.hist(bins="auto",figsize=(20,20), column=['Primary Energy Consumption',
       'Primary Energy Consumption per capita', 'Oil Proved Reserves',
       'Oil Production Barrels', 'Oil Production', 'Oil Consumption Daily',
       'Oil Consumption Total', 'Natural Gas Proved Reserves',
       'Natural Gas Production', 'Natural Gas Consumption',
       'Natural Gas Consumption Mtoe', 'Coal Consumption',
       'Hydroelectricity Generation', 'Hydroelectricity Consumption',
       'Other Renewables Generation', 'Other Renewables Consumption',
       'Electricity Generation', 'Carbon Dioxide Emission'])
plt.show()
df.plot(sharex=True,kind='line',subplots=True, layout=(6,5),figsize=(20,20))
plt.show()
grf=plt.axes()

grf.plot(df['Year'],df['Oil Consumption Total'], color="b", label="Oil")
grf.plot(df['Year'],df['Natural Gas Consumption Mtoe'], color="g", label= "Natural Gas")
grf.plot(df['Year'],df['Coal Consumption'], color="r", label="Coal")
grf.plot(df['Year'],df['Hydroelectricity Consumption'], color="y", label="Hydroelectricity")
grf.plot(df['Year'],df['Renewable Consumption'], color="m", label='Renewable')

grf.set(xlabel='Year', ylabel='Mtoe',title='Consumption')
plt.xlim((1965,2018))
grf.legend()
grf.grid(b=True, which='major', axis='both')
grf.xaxis.set_major_locator(plt.MultipleLocator(5))
grf.yaxis.set_major_locator(plt.MultipleLocator(2.5))


plt.rcParams["figure.figsize"] = [16,9]
plt.show()
fig, ax = plt.subplots()
ax.stackplot(df['Year'],df['Hydroelectricity Consumption'],df['Oil Consumption Total'],df["Natural Gas Consumption Mtoe"]
             ,df['Coal Consumption'],df['Wind Consumption'],df['Other Renewables Consumption'],df['Solar Consumption'], 
             labels=['Hydroelectricity','Oil','Natural Gas','Coal','Wind','Other Renewables','Solar'])
ax.figure.legend(loc='upper left', bbox_to_anchor=(0.1, 0.8))
plt.xlim((1965,2018))
plt.title('Energy Consumption')
ax.set(xlabel='Year', ylabel='Mtoe',title='Consumption')
plt.show()
renov=df[['Year','Hydroelectricity Generation','Solar Generation','Wind Generation','Other Renewables Generation']]
renov.fillna(0,inplace=True);
colors = ['dodgerblue','olivedrab','lightgreen','aqua']

plt.figure(1)

fig1, ax1 = plt.subplots()
ax1.pie(renov.iloc[53,1:6], colors = colors, labels=['Hydroelectricity Generation','Solar Generation','Wind Generation','Other Renewables Generation'], 
        autopct='%1.1f%%')

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

ax1.axis('equal')  
plt.tight_layout()
plt.rcParams["figure.figsize"] = [10,10]

plt.title('Renewables Share of Consumption 2018')
plt.show()
NO_renov=df[['Oil Consumption Total','Coal Consumption','Natural Gas Consumption Mtoe']]
NO_renov.fillna(0,inplace=True)
cols=['Oil','Coal','Natural Gas']

perc = pd.DataFrame(columns = cols)

perc['Oil']=NO_renov['Oil Consumption Total']
perc['Coal']=NO_renov['Coal Consumption']
perc['Natural Gas']=NO_renov['Natural Gas Consumption Mtoe']

perc[cols] = perc[cols].div(perc[cols].sum(axis=1), axis=0).multiply(100)

perc.set_index(df['Year']);
perc.plot(kind='bar', stacked=True, rot=1, figsize=(10, 8), ylim=(0,100),
               title="% Generación de NO renovables")
plt.xticks(range(0,54),df['Year'], rotation=90)
plt.show()
OIL=df[['Oil Proved Reserves','Oil Production Barrels','Oil Production','Oil Consumption Daily','Oil Consumption Total']]
OIL.plot(sharex=True,kind='line',subplots=True, layout=(5,5),figsize=(20,20))
plt.xticks(range(0,54),df['Year'], rotation=90)
plt.show()
diferencia=df['Oil Production']-df['Oil Consumption Total']
ax= plt.axes()
plt.plot(df['Year'],diferencia,color='dodgerblue', linestyle='-')

ax.set(xlabel='Year', ylabel='Mto',title='Diferencia entre Producción y Consumo')
plt.xlim((1965,2018))
plt.grid(b=True, which='major', axis='both')
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
plt.fill_between(df['Year'], diferencia, color='cyan')

GAS_NATURAL=df[['Natural Gas Proved Reserves','Natural Gas Production','Natural Gas Production Mtoe',
                'Natural Gas Consumption','Natural Gas Consumption Mtoe']]
GAS_NATURAL.plot(sharex=True,kind='line',subplots=True, layout=(5,5),figsize=(20,20))
plt.xticks(range(0,54),df['Year'], rotation=90)
plt.show()
sns.pairplot(GAS_NATURAL)
plt.show()
diferencia_GN=df['Natural Gas Production']-df['Natural Gas Consumption']
ax= plt.axes()
plt.plot(df['Year'],diferencia_GN,color='darkgreen', linestyle='-')

ax.set(xlabel='Year', ylabel='Billion cubic meters',title='Diferencia entre Producción y Consumo')
plt.xlim((1965,2018))
plt.grid(b=True, which='major', axis='both')
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
plt.fill_between(df['Year'], diferencia_GN, color='lime')
ax=df.plot(x="Year", y="Electricity Generation",legend= False,color='y')
ax2=ax.twinx()
df.plot(x='Year',y='Carbon Dioxide Emission',ax=ax2,legend=False, color='dimgrey')
ax.figure.legend(loc='upper left', bbox_to_anchor=(0.1, 0.75))

ax.grid(b=True, which='major', axis='both')
ax2.grid(b=False, which='major', axis='both')
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
ax.set( ylabel='Terawatt-hour',title='Emissions & Electricity')
ax2.set( ylabel='Mt of C02')
plt.show()
ax=df.plot.scatter(x='Year',y='Electricity Generation',c='Carbon Dioxide Emission',colormap='jet')
ax.set_facecolor("darkgray")
ax.set(title='Electricity vs Emissions')
ax.grid()
plt.show()
ax=sns.scatterplot(x='Electricity Generation', y='Carbon Dioxide Emission', data=df)
#sns.grid()
sns.set()
GENERACION_Mtoe=df[['Oil Production','Natural Gas Production']]
GENERACION_Twh=df[['Hydroelectricity Generation','Solar Generation','Wind Generation','Other Renewables Generation']]

GENERACION  = pd.concat([GENERACION_Mtoe, GENERACION_Twh*0.085984], axis=1)
GENERACION.tail(5)
fig,ax = plt.subplots()
ax.stackplot(df['Year'],GENERACION["Natural Gas Production"],GENERACION['Oil Production'],GENERACION['Hydroelectricity Generation']
             ,GENERACION['Wind Generation'],GENERACION['Solar Generation'], GENERACION['Other Renewables Generation'],
             labels=['Natural Gas','Oil','Hydroelectricity','Wind','Solar','Other Renewables'])
ax.figure.legend(loc='upper left', bbox_to_anchor=(0.1, 0.8))
plt.xlim((1965,2018))
plt.title('Energy GENERATION')
ax.set(xlabel='Year', ylabel='Mtoe',title='Producción')
plt.show()
CONSUMO_Mtoe=df[['Oil Consumption Total','Natural Gas Consumption Mtoe']]
CONSUMO_Twh=df[['Hydroelectricity Consumption','Solar Consumption','Wind Consumption','Other Renewables Consumption']]

CONSUMO  = pd.concat([CONSUMO_Mtoe, CONSUMO_Twh], axis=1)
CONSUMO.tail(5)
IND_EN= pd.DataFrame(GENERACION.values / CONSUMO.values,columns=['Oil','Natural Gas','Hydroelectricity','Solar','Wind','Other Renewables'])*100


IND_EN.tail(5)
ax= plt.axes()
plt.plot(df['Year'],IND_EN[['Oil','Natural Gas']])

ax.set(xlabel='Year', ylabel='%',title='Independencia Energética')
plt.legend(labels=['Oil','Natural Gas'])
plt.xlim((1965,2018))
plt.grid(b=True, which='major', axis='both')
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.yaxis.set_major_locator(plt.MultipleLocator(250))
PIB=pd.read_excel('/kaggle/input/produccion-energetica-noruega/PIB Noruega.xlsx')

PIB.plot(sharex=True,kind='line',subplots=True, layout=(5,5),figsize=(20,20))
plt.show()
In_En=df['Primary Energy Consumption'].div(PIB['PIB'])*10**9
In_En.columns=['Intensidad Energética']

fig=plt.figure()

ax1=fig.add_subplot()
ax1.plot(df['Year'],In_En, label='Intensidad Energética')
ax1.set_ylabel('Kilogram of Oil Equivalent / US$')
ax1.grid(b=True, which='major', axis='both')
ax1.xaxis.set_major_locator(plt.MultipleLocator(5))
ax1.yaxis.set_major_locator(plt.MultipleLocator(0.25))
ax1.set(title='Intensidad Energética & PIB per capita')

ax2=ax1.twinx()
ax2.plot(df['Year'],PIB['PIB per capita'],label='PIB per capita', color='red')
ax1.figure.legend(loc='upper left', bbox_to_anchor=(0.35, 0.7))

plt.xlim((1965,2018))
plt.show()
ax=plt.axes()

plt.scatter(PIB['PIB per capita'],In_En, label='Intensidad Energética', marker="+", color='green')
plt.grid()
plt.xlabel('US$ per capita')
plt.ylabel('Kilogram of Oil Equivalent / US$')
ax.xaxis.set_major_locator(plt.MultipleLocator(10000))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
