import pandas as pd                

import numpy as np    



import matplotlib.pyplot as plt     

import plotly.express as px       

import plotly.offline as py       

import seaborn as sns             

import plotly.graph_objects as go 

from plotly.subplots import make_subplots

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from matplotlib.pyplot import figure, show



import warnings

warnings.filterwarnings('ignore')

from IPython.display import HTML

import matplotlib.colors as mc

import colorsys

from random import randint

import re

import missingno as msno





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/coronavirusdataset/'



case = p_info = pd.read_csv(path+'Case.csv')

p_info = pd.read_csv(path+'PatientInfo.csv')

p_route = pd.read_csv(path+'PatientRoute.csv')

time = pd.read_csv(path+'Time.csv')

t_age = pd.read_csv(path+'TimeAge.csv')

t_gender = pd.read_csv(path+'TimeGender.csv')

t_provin = pd.read_csv(path+'TimeProvince.csv')

region = pd.read_csv(path+'Region.csv')

weather = pd.read_csv(path+'Weather.csv')

search = pd.read_csv(path+'SearchTrend.csv')

floating = pd.read_csv(path+'SeoulFloating.csv')

policy = pd.read_csv(path+'Policy.csv')
p_info.head()
p_info.isnull().sum()
for col in p_info.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (p_info[col].isnull().sum() / p_info[col].shape[0]))

    print(msg)
msno.matrix(df=p_info.iloc[:, :], figsize=(8,8), color=(0.7,0.9,0.2))

f, ax = plt.subplots(1,2, figsize=(18,8))

p_info['state'].value_counts().plot.pie(explode=[0, 0.1,0], autopct='%1.1f%%', ax=ax[0], shadow=False)

ax[0].set_title('Pie plot -state')

ax[0].set_ylabel('')

sns.countplot('state', data=p_info, ax=ax[1])

ax[1].set_title("Count plot- state")



plt.show()
order = ['0s','10s','20s','30s','40s','50s','60s','70s','80s','90s','100s']

figure(figsize=(20,20))

graph= sns.countplot('age', hue='state', data = p_info, order=order)

for p in graph.patches:

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")

show()
print('Mortality rate within the age group')

print('Mortality rate of 30s: {:.2f}%'.format(1*100/(1+169+282)))

print('Mortality rate of 40s: {:.2f}%'.format(2*100/(2+150+308)))

print('Mortality rate of 50s: {:.2f}%'.format(7*100/(7+375+218)))

print('Mortality rate of 60s: {:.2f}%'.format(12*100/(12+190+203)))

print('Mortality rate of 70s: {:.2f}%'.format(19*100/(19+81+104)))

print('Mortality rate of 80s: {:.2f}%'.format(23*100/(23+84+49)))

print('Mortality rate of 90s: {:.2f}%'.format(7*100/(7+13+25)))

p_info['sex'].fillna('NaN', inplace=True)
sns.countplot('sex', hue='state', data = p_info)

ax[1].set_title('Sex: State')

plt.show
pd.crosstab(p_info['sex'], p_info['state'], margins = True).style.background_gradient(cmap='summer_r')
print('Mortality rate by sex')

print('Mortality rate of male: {:.2f}%'.format(46*100/(1469)))

print('Mortality rate of female: {:.2f}%'.format(25*100/(1865)))
p_info['disease'].fillna('False', inplace=True)
p_info.disease.describe()
f, ax = plt.subplots(1,2, figsize=(18,8))

p_info['disease'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=False)

ax[0].set_title('Pie plot -disease')

ax[0].set_ylabel('')

graph= sns.countplot('disease', data=p_info, ax=ax[1])

ax[1].set_title("Count plot- disease")

for p in graph.patches:

    height = p.get_height()

    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")

plt.show()
pd.crosstab(p_info['disease'], p_info['state'], margins = True).style.background_gradient(cmap='summer_r')
print('Mortality rate by presence of underlying disease')

print('Mortality rate of patients with underlying disease: {:.2f}%'.format(18*100/(18)))

print('Mortality rate of patients without underlying disease: {:.2f}%'.format(53*100/(3501)))
figure(figsize=(20,20))

graph= sns.countplot('province', hue='state', data = p_info)

ax[1].set_title('Province: State')

graph.set_xticklabels(graph.get_xticklabels(), rotation=45)

plt.show
pd.crosstab(p_info['province'], p_info['state'], margins = True).style.background_gradient(cmap='summer_r')
print('Mortality rate of patients from Daegu: {:.2f}%'.format(20*100/(63)))

print('Mortality rate of patients from Busan: {:.2f}%'.format(3*100/(141)))



print('Mortality rate of patients from Gyeongsangbuk-do: {:.2f}%'.format(40*100/(1230)))

print('Mortality rate of patients from Seoul: {:.2f}%'.format(4*100/(714)))
