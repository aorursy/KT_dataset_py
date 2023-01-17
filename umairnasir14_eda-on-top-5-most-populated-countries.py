# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), Will be used for visualization as well

# Visualization libraries

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install dexplot
!pip install chart_studio

import chart_studio.plotly as py
import dexplot as dxp 
df_China = pd.read_excel(open('../input/population-growth-of-top-25-countries/World Population.xlsx','rb'),sheet_name='China')
df_India = pd.read_excel(open('../input/population-growth-of-top-25-countries/World Population.xlsx','rb'),sheet_name='India')
df_USA = pd.read_excel(open('../input/population-growth-of-top-25-countries/World Population.xlsx','rb'),sheet_name='USA')
df_Indonasia = pd.read_excel(open('../input/population-growth-of-top-25-countries/World Population.xlsx','rb'),sheet_name='Indonasia')
df_Pakistan = pd.read_excel(open('../input/population-growth-of-top-25-countries/World Population.xlsx','rb'),sheet_name='Pakistan')

df_China.head()


df_China.columns = df_China.columns.str.replace(' ', '_')
df_China=df_China.rename(columns={'Yearly_%\nChange':'Yearly_%_Change','Yearly\nChange':'Yearly_Change',
                        "Country's_Share_of\nWorld_Pop":"Country's_Share_of_World_Pop",
                        'Urban\nPop_%':'Urban_Pop_%','\nGlobal_Rank':'Global_Rank'})

df_India.columns = df_India.columns.str.replace(' ', '_')
df_India=df_India.rename(columns={'Yearly_%\nChange':'Yearly_%_Change','Yearly\nChange':'Yearly_Change',
                        "Country's_Share_of\nWorld_Pop":"Country's_Share_of_World_Pop",
                        'Urban\nPop_%':'Urban_Pop_%','\nGlobal_Rank':'Global_Rank'})

df_USA.columns = df_USA.columns.str.replace(' ', '_')
df_USA=df_USA.rename(columns={'Yearly_%\nChange':'Yearly_%_Change','Yearly\nChange':'Yearly_Change',
                        "Country's_Share_of\nWorld_Pop":"Country's_Share_of_World_Pop",
                        'Urban\nPop_%':'Urban_Pop_%','\nGlobal_Rank':'Global_Rank'})

df_Indonasia.columns = df_Indonasia.columns.str.replace(' ', '_')
df_Indonasia=df_Indonasia.rename(columns={'Yearly_%\nChange':'Yearly_%_Change','Yearly\nChange':'Yearly_Change',
                        "Country's_Share_of\nWorld_Pop":"Country's_Share_of_World_Pop",
                        'Urban\nPop_%':'Urban_Pop_%','\nGlobal_Rank':'Global_Rank'})

df_Pakistan.columns = df_Pakistan.columns.str.replace(' ', '_')
df_Pakistan=df_Pakistan.rename(columns={'Yearly_%\nChange':'Yearly_%_Change','Yearly\nChange':'Yearly_Change',
                        "Country's_Share_of\nWorld_Pop":"Country's_Share_of_World_Pop",
                        'Urban\nPop_%':'Urban_Pop_%','\nGlobal_Rank':'Global_Rank'})

df_China.head()

frames = [df_China, df_India, df_USA, df_Indonasia, df_Pakistan]
df_concat= pd.concat(frames)
df_concat['above1300m'] = ['above1300m'if i >=1300000000 else 'below1300m'for i in df_concat.Population]

arr = df_concat['Median_Age'].unique()
print("Median of Age: ",np.median(arr))


df_concat['Median_of_Age'] = ['higher_24.9'if i >=24.9 else 'below_24.9'for i in df_concat['Median_Age']]
arr = df_concat['Urban_Population'].unique()
print("Median of Urban Population: ",np.median(arr))

df_concat['Median_of_Urban_Population'] = ['higher_145948933.5'if i >=145948933.5 else 'below_145948933.5'for i in df_concat['Urban_Population']]
df_concat.info()
df_concat.head()
fig = plt.figure()
spec = gridspec.GridSpec(ncols=2, nrows=3,
                         width_ratios=[4,4])



ax0 = fig.add_subplot(spec[0])
ax0.set_ylabel('Yearly_change')
ax0.plot('Year','Yearly_Change', data = df_China)

ax1 = fig.add_subplot(spec[1])
ax1.plot('Year','Yearly_Change', data = df_India)

ax2 = fig.add_subplot(spec[2])
ax2.plot('Year','Yearly_Change', data = df_USA)

ax3 = fig.add_subplot(spec[3])
ax3.plot('Year','Yearly_Change', data = df_Indonasia)

ax4 = fig.add_subplot(spec[4])
ax4.set_xlabel('Year')
ax4.plot('Year','Yearly_Change', data = df_Pakistan)




plt.show()


def drawAnyAmount(a,i):
    sns.countplot(x=df_concat[a],  palette="Set2",ax=axs[i])
    axs[i].set_title(a, color='blue', fontsize=15)
    

a=['Median_of_Age','above1300m', 'Median_of_Urban_Population']
iterr= 0

numToPlot = len(a)
fig, axs =plt.subplots(ncols=numToPlot)
plt.subplots_adjust(right=2, wspace = 0.5)
for i in a:
    drawAnyAmount(i,iterr)
    iterr +=1
fig, ax = plt.subplots(figsize=(10,10))

sns.swarmplot(x='above1300m',y='Population',hue='Median_of_Age',data=df_concat, ax=ax)
plt.show()
fig, ax = plt.subplots(figsize=(10,10))
sns.swarmplot(x='Median_of_Urban_Population',y='Population',hue='Median_of_Age',data=df_concat, ax=ax)
plt.show()
plt.style.use("classic")
sns.distplot(df_concat['Fertility_Rate'], color='blue')
plt.xlabel("Fertility_Rate")
plt.ylabel("Count")
plt.show()
df_China[['Population', 'Urban_Population']].plot.box(figsize=(10,10), colormap =  'autumn')
df_USA[['Population', 'Urban_Population']].plot.box(figsize=(10,10), colormap =  'RdYlGn')
df_India[['Population', 'Urban_Population']].plot.box(figsize=(10,10), colormap =  'BrBG')
df_Indonasia[['Population', 'Urban_Population']].plot.box(figsize=(10,10), colormap =  'PRGn')
df_Pakistan[['Population', 'Urban_Population']].plot.box(figsize=(10,10), colormap =  'winter_r')
dxp.count('Migrants_(net)',df_USA, split='Fertility_Rate',normalize=True,figsize=(8,6),size=0.9,stacked=True)
dxp.count('Median_Age',df_USA, split='Fertility_Rate',normalize=True,figsize=(8,6),size=0.9,stacked=True)
dxp.bar(x='Urban_Population', y='Density_(P/Km²)', data=df_USA, figsize=(10,4), aggfunc='median')
dxp.bar(x='Urban_Population', y='Yearly_Change', data=df_USA, figsize=(10,4),  aggfunc='median')
dxp.line(x='Fertility_Rate', y='Yearly_Change', data=df_USA, figsize = (10,4), aggfunc='median')
dxp.line(x='Fertility_Rate', y='Density_(P/Km²)', data=df_USA, figsize = (10,4), aggfunc='median')
dxp.line(x='Urban_Population', y='Density_(P/Km²)', data=df_USA, figsize=(10,4), aggfunc='median', orientation='h')
dxp.box(x='Year', y='Urban_Pop_%', data=df_USA,  figsize=(10,4))
dxp.box(x='Migrants_(net)', y='Year', data=df_USA,orientation='v',figsize = (12,4),
        split='Global_Rank', split_order='top 2')
dxp.kde(x='Year', y='Population', data=df_USA, figsize=(4,4))
