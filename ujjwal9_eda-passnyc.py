import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")
exp = pd.read_csv('../input/2016 School Explorer.csv')
shsat = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
exp.head()
shsat.head()
exp['School Income Estimate'] = exp['School Income Estimate'].replace('[\$,]', '', regex=True).astype(float)
exp['Trust %'] = exp['Trust %'].replace('[\%,]', '', regex=True).astype(float)
exp['Student Attendance Rate'] = exp['Student Attendance Rate'].replace('[\%,]', '', regex=True).astype(float)
df = exp.groupby('School Name').agg({'School Income Estimate':sum, 'Trust %':'mean', 'Student Attendance Rate':'mean'})
df = df.reset_index(level=[0])
df.iplot(kind='bubble', x='Student Attendance Rate', y='Trust %', size='School Income Estimate', title='School Income Estimate based on Attendance and Trust %',
             yTitle='Mean Trust %', xTitle='Mean Student Attendance Rate')
plt.figure(figsize=(12,8))
temp = exp.groupby('Community School?')['Trust %'].mean()
temp.iplot(kind='bar', yTitle='mean trust %', xTitle="Community School or not", title='avg trust % among community and non-community schools')
df = exp.groupby('School Name').agg({'School Income Estimate':sum, 'Trust %':'mean', 'Student Attendance Rate':'mean'})
df = df.reset_index(level=[0])
df.iplot(kind='bubble', x='Student Attendance Rate', y='Trust %', size='School Income Estimate', title='School Income Estimate based on Attendance and Trust %',
             yTitle='Mean Trust %', xTitle='Mean Student Attendance Rate')
