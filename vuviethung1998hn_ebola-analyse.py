import pandas as pd

import numpy as np

import matplotlib as matplot

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

%matplotlib inline
eb_df = pd.read_csv('../input/country_timeseries.csv')
ebola_melt = pd.melt(eb_df, id_vars = ['Date', 'Day'], var_name='type_country', value_name='counts')

ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

ebola_melt['type'] = ebola_melt.str_split.str.get(0)
ebola_melt['country'] = ebola_melt.str_split.str.get(1)
ebola_melt['Dmy'] = ebola_melt.Date.str.split('/')

ebola_melt['D'] = ebola_melt.Dmy.str.get(1)

ebola_melt['M'] = ebola_melt.Dmy.str.get(0)

ebola_melt['Y'] = ebola_melt.Dmy.str.get(2)
ebola_melt.drop(['type_country', 'str_split','Dmy'], axis = 1)
ebola_tidy = ebola_melt.pivot_table(values='counts', index=['D','M','Y', 'Day', 'country'], columns='type', aggfunc=np.sum)

ebola_reset = ebola_tidy.reset_index()
ebola_reset.rename(columns={'country':'Country'}, inplace=True) 

print(ebola_reset.head())
ebola_reset['D']=pd.to_numeric(ebola_reset['D'], errors='coerce')

ebola_reset['M']=pd.to_numeric(ebola_reset['M'], errors='coerce')

ebola_reset['Y']=pd.to_numeric(ebola_reset['Y'], errors='coerce')
df_final = ebola_reset.sort_values('Cases', ascending=False).drop_duplicates('Country')

frames_country = {}
list_country = ebola_reset.Country.unique()

print(list_country)
for ctry in list_country:

    df = ebola_reset[(ebola_reset['Country'] == ctry)].sort_values(['Cases','Day'], ascending=True)

    frames_country[ctry] = df

print(frames_country['UnitedStates'])
x = frames_country['UnitedStates'].Day

N = len(x)

y = frames_country['UnitedStates'].Cases

z = frames_country['UnitedStates'].Deaths 

ind = np.arange(N)#the x location for groups

width = 0.35 # the width of the bars

p1 = plt.bar(ind, y,  width)

p2 = plt.bar(ind, z, width)

plt.ylabel('Cases')

plt.title('Percentage of Deaths over Cases in United States')

plt.legend((p1[0],p2[0]),('Cases','Deaths'))

plt.show()
x = frames_country['Senegal'].Day

N = len(x)

y = frames_country['Senegal'].Cases

z = frames_country['Senegal'].Deaths 

ind = np.arange(N)#the x location for groups

width = 0.35 # the width of the bars

p1 = plt.bar(ind, y,  width)

p2 = plt.bar(ind, z, width)

plt.ylabel('Cases')

plt.title('Percentage of Deaths over Cases in Senegal')

plt.legend((p1[0],p2[0]),('Cases','Deaths'))

plt.show()
x = frames_country['Guinea'].Day

N = len(x)

y = frames_country['Guinea'].Cases

z = frames_country['Guinea'].Deaths 

ind = np.arange(N)#the x location for groups

width = 0.35 # the width of the bars

p1 = plt.bar(ind, y,  width)

p2 = plt.bar(ind, z, width)

plt.ylabel('Cases')

plt.title('Percentage of Deaths over Cases in Guinea')

plt.legend((p1[0],p2[0]),('Cases','Deaths'))

plt.show()
x = frames_country['Liberia'].Day

N = len(x)

y = frames_country['Liberia'].Cases

z = frames_country['Liberia'].Deaths 

ind = np.arange(N)#the x location for groups

width = 0.35 # the width of the bars

p1 = plt.bar(ind, y,  width)

p2 = plt.bar(ind, z, width)

plt.ylabel('Cases')

plt.title('Percentage of Deaths over Cases in Liberia')

plt.legend((p1[0],p2[0]),('Cases','Deaths'))

plt.show()
x = frames_country['Mali'].Day

N = len(x)

y = frames_country['Mali'].Cases

z = frames_country['Mali'].Deaths 

ind = np.arange(N)#the x location for groups

width = 0.35 # the width of the bars

p1 = plt.bar(ind, y,  width)

p2 = plt.bar(ind, z, width)

plt.ylabel('Cases')

plt.title('Percentage of Deaths over Cases in Mali')

plt.legend((p1[0],p2[0]),('Cases','Deaths'))

plt.show()
x = frames_country['Nigeria'].Day

N = len(x)

y = frames_country['Nigeria'].Cases

z = frames_country['Nigeria'].Deaths 

ind = np.arange(N)#the x location for groups

width = 0.35 # the width of the bars

p1 = plt.bar(ind, y,  width)

p2 = plt.bar(ind, z, width)

plt.ylabel('Cases')

plt.title('Percentage of Deaths over Cases in Nigeria')

plt.legend((p1[0],p2[0]),('Cases','Deaths'))

plt.show()
x = frames_country['Spain'].Day

N = len(x)

y = frames_country['Spain'].Cases

z = frames_country['Spain'].Deaths 

ind = np.arange(N)#the x location for groups

width = 0.35 # the width of the bars

p1 = plt.bar(ind, y,  width)

p2 = plt.bar(ind, z, width)

plt.ylabel('Cases')

plt.title('Percentage of Deaths over Cases in Spain')

plt.legend((p1[0],p2[0]),('Cases','Deaths'))

plt.show()
x = frames_country['SierraLeone'].Day

N = len(x)

y = frames_country['SierraLeone'].Cases

z = frames_country['SierraLeone'].Deaths 

ind = np.arange(N)#the x location for groups

width = 0.35 # the width of the bars

p1 = plt.bar(ind, y,  width)

p2 = plt.bar(ind, z, width)

plt.ylabel('Cases')

plt.title('Percentage of Deaths over Cases in SierraLeone')

plt.legend((p1[0],p2[0]),('Cases','Deaths'))

plt.show()