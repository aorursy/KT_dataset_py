import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
df.head(10)
df.info()
msno.bar(df, color='green')
sal = df['Salary Estimate'].apply(lambda x:x.split()[0])
df['sal_min'] = sal.apply(lambda x: x.split('-')[0][1:-1])
df['sal_max'] = sal.apply(lambda x: x.split('-')[1][1:-1])
df[['sal_min', 'sal_max']].head()
#change salary datatype
df['sal_min'].replace('', '0', inplace=True)
df['sal_min'] = df['sal_min'].astype('float')
df['sal_max'].replace('', '0', inplace=True)
df['sal_max'] = df['sal_max'].astype('float')
df['Easy Apply'].replace({'-1': '0', 'True': '1'}, inplace=True)
df['Competitors'].replace('-1', 'No Competitor Found', inplace=True)
df['Location_Abb'] = df['Location'].apply(lambda x: x.split(',')[1])
df['minimum_company_size'] = df['Size'].apply(lambda x: x.split()[0])
df['minimum_company_size'].replace('10000+', '10000', inplace=True)
df.head()
df['Revenue'].unique()
rev = df.groupby('Revenue').count().sort_values(ascending=False, by='Job Title')
rev = rev.reset_index()
plt.figure(figsize=(14,8))
sns.barplot(x='Revenue',y='Job Title', data=rev, palette='OrRd_r')
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
sns.despine(left=True)
Ind_10 = df.groupby('Industry').sum().sort_values(ascending=False, by='sal_min').reset_index()['Industry'][0:16]
filtered_df = df[df['Industry'].isin(Ind_10)]
filtered_df = filtered_df[filtered_df != '-1']

plt.figure(figsize=(14,8))
#sns.stripplot(x='Industry', y='sal_min', data=filtered_df, palette='coolwarm')
sns.boxplot(x='Industry', y='sal_min', data=filtered_df, palette='magma')

plt.xticks(
    rotation=70, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
sns.despine(left=True)
plt.figure(figsize=(14,8))
sns.scatterplot(x='minimum_company_size', y='sal_min', data=df, hue=df['Rating'], palette='coolwarm', s=100)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.despine(left=True)
plt.figure(figsize=(14,8))
sns.stripplot(x='minimum_company_size', y='sal_min', data=df, hue=df['Easy Apply'], palette='magma', s=6)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.despine(left=True)
Ind_10 = df.groupby('Sector').sum().sort_values(ascending=False, by='sal_min').reset_index()['Sector'][0:11]
filtered_df_1 = df[df['Sector'].isin(Ind_10)]
filtered_df_1 = filtered_df_1[filtered_df_1 != '-1']

plt.figure(figsize=(14,8))
#sns.stripplot(x='Industry', y='sal_min', data=filtered_df, color='green')
sns.boxplot(x='Sector', y='sal_min', data=filtered_df_1, palette='OrRd')

plt.xticks(
    rotation=35, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
sns.despine(left=True)
Ind_10 = df.groupby('Job Title').sum().sort_values(ascending=False, by='sal_min').reset_index()['Job Title'][0:10]
filtered_df_2 = df[df['Job Title'].isin(Ind_10)]
filtered_df_2 = filtered_df_2[filtered_df_2 != '-1']

plt.figure(figsize=(14,8))
#sns.stripplot(x='Industry', y='sal_min', data=filtered_df, color='green')
sns.boxplot(x='Job Title', y='sal_min', data=filtered_df_2, palette='GnBu_r')

plt.xticks(
    rotation=35, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
sns.despine(left=True)
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objs as go

df['Location_Abb'].replace({' Arapahoe': 'CO'}, inplace=True)
df_loc = df.groupby('Location_Abb').count().reset_index()
df_loc = pd.DataFrame(df_loc)
df_loc['Location_Abb'] = df_loc['Location_Abb'].apply(lambda x: x.split()[0])
data = dict(type='choropleth',colorscale='RdBu_r', locations = df_loc['Location_Abb'], locationmode = 'USA-states', z= df_loc['Job Title'], colorbar={'title':'Scale'},  marker = dict(line=dict(width=0))) 
layout = dict(title = 'Data Analyst Job Market!', geo = dict(scope='usa')) # , showlakes=True, lakecolor = 'grey'))
Choromaps2 = go.Figure(data=[data], layout=layout)
iplot(Choromaps2)