import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
color = sns.color_palette()
from numpy import array
from matplotlib import cm
from scipy.misc import imread
import base64
from sklearn import preprocessing
#from mpl_toolkits.basemap import Basemap
from wordcloud import WordCloud, STOPWORDS
import plotly.plotly as py1
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools


import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_d5 = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
df_exp = pd.read_csv('../input/2016 School Explorer.csv')
desc = pd.DataFrame({'Dataset':['Registrations and Testers','School Explorer'],
             'Datapoints':[df_d5.shape[0],df_exp.shape[0]],
             'Features':[df_d5.shape[1],df_exp.shape[1]]})
desc = desc[['Dataset','Datapoints','Features']]
desc
df_d5.head(3)
df_exp.head(3)
df_d5.dtypes
plt.figure(figsize=(14,10))
plt.barh(df_d5['School name'].value_counts().index[::-1], 
       df_d5['School name'].value_counts()[::-1],
       color=sns.color_palette('viridis'))
plt.xlabel('Counts')
plt.ylabel('Schools')
plt.title('School Distribution', size = 15)
plt.tight_layout()
plt.figure(figsize=(10,6))

plt.bar(df_d5['Year of SHST'].value_counts().index, 
        df_d5['Year of SHST'].value_counts(),
        color=sns.color_palette('viridis'))
plt.xlabel('Year')
plt.xticks([2013,2014,2015,2016])
plt.ylabel('Counts')
plt.title('Year of SHST Distribution', size = 18)
plt.tight_layout()
plt.figure(figsize=(4,4))
plt.pie(df_d5['Grade level'].value_counts(),radius=2,
       colors=sns.color_palette('viridis'),
       labeldistance = 1.1);

g8_pct = (df_d5['Grade level'].value_counts().values[0]/len(df_d5)).round(3)
g9_pct = (df_d5['Grade level'].value_counts().values[1]/len(df_d5)).round(3)
plt.text(-3.5,0, 'Grade Level', size = 18)
plt.text(-3.5,-0.2, 'Distribution', size = 18)
plt.text(-0.5,0.75,'Grade 8', color = 'white', size = 20)
plt.text(-0.5,0.5,str(g8_pct)+'%', color = 'white', size = 20)

plt.text(-0.2,-0.9,'Grade 9', color = 'white', size = 20)
plt.text(-0.2,-1.2,str(g9_pct)+'%', color = 'white', size = 20)
plt.text(-0.2,-0.9,'Grade 9', color = 'white', size = 20)
plt.tight_layout()
pd.DataFrame(df_d5['Enrollment on 10/31'].describe())
#df_d5.groupby('School name')['Enrollment on 10/31'].sum()

plt.figure(figsize=(14,10))
plt.barh(df_d5.groupby('School name')['Enrollment on 10/31'].sum().index, 
       df_d5.groupby('School name')['Enrollment on 10/31'].sum(),
       color=sns.color_palette('viridis'))
plt.xlabel('Enrollments on 10/31')
plt.ylabel('School')
plt.title('Enrollments on 10/31 School Distribution', size = 15)
plt.tight_layout()
pd.DataFrame(df_d5['Number of students who registered for the SHSAT'].describe())
plt.figure(figsize=(14,10))
plt.barh(df_d5.groupby('School name')['Number of students who registered for the SHSAT'].sum().index, 
       df_d5.groupby('School name')['Number of students who registered for the SHSAT'].sum(),
       color=sns.color_palette('viridis'))
plt.xlabel('Number of students who registered for the SHSAT')
plt.ylabel('School')
plt.title('Number of students who registered for the SHSAT School Distribution', size = 15)
plt.tight_layout()
pd.DataFrame(df_d5['Number of students who took the SHSAT'].describe())
plt.figure(figsize=(14,10))
plt.barh(df_d5.groupby('School name')['Number of students who took the SHSAT'].sum().index, 
       df_d5.groupby('School name')['Number of students who took the SHSAT'].sum(),
       color=sns.color_palette('plasma'))
plt.xlabel('Number of students who took the SHSAT')
plt.ylabel('School')
plt.title('Number of students who took the SHSAT  School Distribution', size = 15)
plt.tight_layout()
df_d5['Diff between registered and takers'] = df_d5['Number of students who \
registered for the SHSAT'] - df_d5['Number of students who took the SHSAT']
pd.DataFrame(df_d5['Diff between registered and takers'].describe())
plt.figure(figsize=(14,10))
plt.barh(df_d5.groupby('School name')['Diff between registered and takers'].sum().index, 
       df_d5.groupby('School name')['Diff between registered and takers'].sum(),
       color=sns.color_palette('plasma'))
plt.xlabel('Difference between registered and takers')
plt.ylabel('School')
plt.title('Difference between registered and takers', size = 15)
plt.tight_layout()
