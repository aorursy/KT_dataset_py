import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import iplot

from scipy.stats import pearsonr

import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std

from sklearn.preprocessing import scale

from sklearn.preprocessing import normalize

import scipy.cluster.hierarchy as shc

from scipy import stats

sns.set_style("darkgrid")

plt.style.use("fivethirtyeight")

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/world-happiness/2019.csv')
df.columns
df.head(10)
df.shape
df.min()
df.style.background_gradient(cmap='Blues')
df.info()
df.describe().style.background_gradient(cmap='Blues')
df.isna().sum()
df.corr().style.background_gradient(cmap="Greens")
plt.figure(figsize=(20,8))

sns.heatmap(df.corr(), annot=True);
df.rename(columns = {'Overall rank':'RANK', 'Country or region':'REGION','GDP per capita':'GDP','Social support':'SCOICL','Healthy life expectancy'

                    :'HIE','Freedom to make life choices':'FTMLC','Perceptions of corruption':'CORUP'}, inplace = True)
df.columns
df1=df[['Score','RANK','GDP','CORUP']]
df1.head(3)
cormat=df1.corr()
sns.heatmap(cormat, annot=True);
differ=df['Score'].max()-df['Score'].min()

z=round(differ/3,3)

low=df['Score'].min()+z

mid=low+z
cat=[]

for i in df.Score:

    if(i>0 and i<low):

        cat.append('Low')

    elif(i>low and i<mid):

         cat.append('Mid')

    else:

         cat.append('High')

df['Category']=cat  
color = (df.Category == 'High' ).map({True: 'background-color: lightblue',False:'background-color: red'})

df.style.apply(lambda s: color)
fig = px.bar(df, x='REGION', y='Score',

             hover_data=['RANK', 'GDP', 'SCOICL', 'HIE', 'FTMLC'], color='GDP')

fig.show()
ndf=pd.pivot_table(df, index = 'REGION', values=["Score","SCOICL"])

ndf["Score"]=ndf["Score"]/max(ndf["Score"])

ndf["SOCIAL"]=ndf["SCOICL"]/max(ndf["SCOICL"])

sns.jointplot(ndf.SCOICL,ndf.Score,kind="kde",height=10,space=0.5)

plt.savefig('graph.png')

plt.show()
df.columns
fig = px.scatter_matrix(df,dimensions=['RANK', 'GDP', 'SCOICL', 'HIE', 'FTMLC','CORUP'],color='Category')

fig.show()
y = df['GDP']

x =  df[['RANK', 'SCOICL', 'HIE', 'FTMLC','CORUP']]

model = sm.OLS(y, x)

results = model.fit()

print(results.summary())
print('Parameters: ', results.params)

print('Standard errors: ', results.bse)
data = dict(type = 'choropleth', 

           locations = df['REGION'],

           locationmode = 'country names',

           colorscale='RdYlGn',

           z = df['Score'], 

           text = df['REGION'],

           colorbar = {'title':'Score'})



layout = dict(title = 'Geographical Visualization of Happiness Score', geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))



choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
df2=df.drop(['REGION', 'RANK','Category'], axis=1)
df2.head()
fig = ff.create_dendrogram(df2, color_threshold=1)

fig.update_layout(width=2000, height=500)

fig.show()