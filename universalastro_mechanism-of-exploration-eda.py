# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
print("Shape of train and test",train.shape, test.shape)

print("No. of unique cp_type: ",train['cp_type'].nunique(),train['cp_type'].unique())
print("No, of unique cp_dose: ",train['cp_dose'].nunique(), train['cp_dose'].unique())
print("No, of unique cp_time (hours): ",train['cp_time'].nunique(), train['cp_time'].unique())
feature_list = [feature for feature in train.columns if feature.startswith('c-') or feature.startswith('g-')]
print("No. of Protein columns: ",len(feature_list))
train.head()
train.describe(include='all').T
fig = px.bar(t1,x='cp_time',  y='count', color='dataset', barmode='group',width=600, height=400)
fig.update_xaxes(title_text='Duration of treatment (in hours)')
fig.update_yaxes(title_text='Number of drugs')

fig.show()
time_train =train.groupby(['cp_time'])['cp_time'].agg({'count'}).reset_index().rename(columns={'id':'count'})
time_train['dataset'] ='train'

time_test =test.groupby(['cp_time'])['cp_time'].agg({'count'}).reset_index().rename(columns={'id':'count'})
time_test['dataset'] ='test'

dose_train =train.groupby(['cp_dose'])['cp_dose'].agg({'count'}).reset_index().rename(columns={'id':'count'})
dose_train['dataset'] ='train'

dose_test =test.groupby(['cp_dose'])['cp_dose'].agg({'count'}).reset_index().rename(columns={'id':'count'})
dose_test['dataset'] ='test'

type_train =train.groupby(['cp_type'])['cp_type'].agg({'count'}).reset_index().rename(columns={'id':'count'})
type_train['dataset'] ='train'

type_test =test.groupby(['cp_type'])['cp_type'].agg({'count'}).reset_index().rename(columns={'id':'count'})
type_test['dataset'] ='test'

t1 = pd.concat([time_train, time_test])
t2 = pd.concat([dose_train, dose_test])
t3 = pd.concat([type_train, type_test])

# Plot Categorical variables
fig1 = px.bar(t1,x='cp_time',  y='count', color='dataset', barmode='group',width=600, height=400)
fig1.update_xaxes(title_text='Duration of treatment (in hours)')
fig1.update_yaxes(title_text='Number of drugs')
fig1.show()

fig3 = px.bar(t3,x='cp_type',  y='count', color='dataset', barmode='group',width=600, height=400)
fig3.update_xaxes(title_text='Type of treatment')
fig3.update_yaxes(title_text='Number of drugs')
fig3.show()

fig2 = px.bar(t2,x='cp_dose',  y='count', color='dataset', barmode='group',width=600, height=400)
fig2.update_xaxes(title_text='Dosage of treatment')
fig2.update_yaxes(title_text='Number of drugs')
fig2.show()

test.info()
random_numbers = np.random.randint(4,876,12)
f, axes = plt.subplots(4, 3, figsize=(18, 18), sharey=False)
colors = ['mediumslateblue','red','olive','gold','navy','magenta','darkcyan','tomato','rebeccapurple','green','brown','mediumvioletred']
row = 0
col=0
for n in range(12):  
    if n%3 == 0 and n!=0:
        row += 1
        col = 0
    else:
        if col == 2:
            col=0
        else:    
            col += 1
    col_name = train.columns[random_numbers[n]]
    c = colors[n]    
    sns.distplot(train[col_name] , color=c, ax=axes[row, col], bins=20,kde=True,rug=True, hist=True)
for r in range(12):

    fig = px.histogram(train, x=train.columns[random_numbers[r]], histfunc='sum', height=300, title='Histogram Chart')
    fig.show()
random_numbers_50 = np.random.randint(4,876,50)
cols_name = [cols for cols in train.columns[random_numbers_50]]
matrix = train[cols_name].corr()

f = plt.figure(figsize=(20, 16))
sns.heatmap(matrix , linewidths=.5, cmap="YlGnBu")
sor = train.corr().abs().unstack().sort_values(ascending=False,kind="quicksort")
print("Correlation Matrix - Shape",sor.shape)
hundred_bool = (sor != 1.000000)
hb = sor[hundred_bool]
print("Correlation Matrix after removing 1 - Shape",hb.shape)
ninety_bool = (hb >= 0.91)
nb = hb[ninety_bool]
print("Correlation Matrix above .90 - Shape",nb.shape)
nb[:5]
corr_cols = ['c-2','c-4','c-6','c-13','c-26','c-33','c-38','c-42','c-52','c-55','c-62','c-63','c-66','c-73','c-82','c-90','c-94']
len(corr_cols)
high_corr = train[corr_cols].corr()
high_corr
sns.heatmap(high_corr , linewidths=.5, cmap=sns.light_palette("navy", reverse=False) )#cmap="YlGnBu")
scored.columns
scored.head()
drug_sums = scored.drop(['sig_id'],axis=1).sum().sort_values(ascending=False).reset_index()#.rename(columns={'index','protein'})

fig1 = px.bar(drug_sums[:50],  x='index',  y=0, height=900, width=1400,labels={'index':'Protein','0':'Frequency'},title="Maximum Protein across drugs")
fig1.show()
fig2 = px.bar(drug_sums[50:],  x='index',  y=0, height=900, width=1400,labels={'index':'Protein','0':'Frequency'},title="Minimum Protein across drugs")
fig2.show()