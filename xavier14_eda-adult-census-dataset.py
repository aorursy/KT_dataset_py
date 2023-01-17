from plotly.offline import init_notebook_mode, iplot_mpl, download_plotlyjs, plot, iplot

import plotly_express as px

import plotly.figure_factory as ff

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings('ignore')

init_notebook_mode(connected=True)

import pandas_profiling

import statsmodels.formula.api as sm

import missingno as msno

from sklearn.preprocessing import LabelEncoder

from statsmodels.compat import lzip

import statsmodels.api as sm

from sklearn.preprocessing import scale

import pandas as pd
data = pd.read_csv("../input/adult-census-income/adult.csv")
data.info()
data.head()
df=data.copy()
px.histogram(df, x="fnlwgt",nbins=int(np.sqrt(len(df.fnlwgt))),

             marginal="box",title='Histogram & Box-Plot: fnlwgt',

             template='ggplot2').update_traces(dict(marker_line_width=1, marker_line_color="black"))
px.histogram(df, x="capital.gain",nbins=int(np.sqrt(len(df['capital.gain']))),

             marginal="box",title='Histogram & Box-Plot: capital-gain',

             template='ggplot2').update_traces(dict(marker_line_width=1, marker_line_color="black"))
px.histogram(df, x="capital.loss",nbins=int(np.sqrt(len(df['capital.loss']))),

             marginal="box",title='Histogram & Box-Plot: capital-loss',

             template='ggplot2').update_traces(dict(marker_line_width=1, marker_line_color="black"))
px.histogram(df, x='hours.per.week',nbins=10,

             marginal="box",title='Histogram & Box-Plot: hour per week',

             template='ggplot2').update_traces(dict(marker_line_width=1, marker_line_color="black"))
px.histogram(df, x='age',nbins=int(np.sqrt(len(df['age']))),

             marginal="box",title='Histogram & Box-Plot: Age',

             template='ggplot2').update_traces(dict(marker_line_width=1, marker_line_color="black"))
df.workclass.value_counts(normalize=True)
df.workclass=df.workclass.replace({"?": "unknown"})
df.workclass.value_counts(normalize=True)
fig = px.treemap(df, path=['sex','race','workclass'],color_discrete_sequence=px.colors.sequential.Magenta)

fig.show()
df.education.value_counts(normalize=True)
fig = px.treemap(df, path=['sex','race','education'],color_discrete_sequence=px.colors.sequential.Teal)

fig.show()
df['marital.status'].value_counts(normalize=True)
fig = px.treemap(df, path=['sex','race','marital.status'],color_discrete_sequence=px.colors.sequential.Emrld)

fig.show()
df['occupation'].value_counts(normalize=True)
df.occupation=df.occupation.replace({' ?': " unknown"})
fig = px.treemap(df, path=['sex','race','occupation'],color_discrete_sequence=px.colors.sequential.Purp)

fig.show()
df['relationship'].value_counts(normalize=True)
fig = px.treemap(df, path=['sex','race','relationship'],color_discrete_sequence=px.colors.sequential.Sunset)

fig.show()
df['race'].value_counts(normalize=True)
fig = px.pie(df, 'race', title='Race Pie Chart distribution').update_traces(hoverinfo='label+value', 

                                                                            textinfo='percent', textfont_size=16)

fig.show()
df['sex'].value_counts(normalize=True)
fig = px.pie(df, 'sex', title='Sex Pie Chart distribution').update_traces(hoverinfo='label+value', textinfo='percent', textfont_size=16)

fig.show()
df['native.country'].unique()
df['native.country']=df['native.country'].replace({" ?": "unknown"})
df['native.country']=df['native.country'].replace({None: "None"})
df['native.country'].value_counts(normalize=True)
fig = px.treemap(df, path=['native.country'])

fig.show()
df.income.value_counts(normalize=True)
df[">50K"]=pd.get_dummies(df.income).iloc[:,1:]
fig = px.pie(df, 'income', title='Pie Chart - Income >50K').update_traces(hoverinfo='value', textinfo='percent+value', textfont_size=20)

fig.show()
fig = px.imshow(df.corr(),x=list(df.corr().columns),y=list(df.corr().columns),width=900, 

                height=700,title='Correlation Matrix', color_continuous_scale=px.colors.diverging.Tropic)

fig.show()
fig, ax = plt.subplots(figsize=(20,6))

fig = sm.graphics.interaction_plot(x=df.race, response=df[">50K"], trace=df.sex,

                                   ax=ax, plottype='both',colors=['coral','lightblue'], markers=['D','^'], ms=16)
fig, ax = plt.subplots(figsize=(20,6))

fig = sm.graphics.interaction_plot(x=df['education.num'], response=df[">50K"], trace=df.sex,

                                   ax=ax, plottype='both',colors=['coral','lightblue'], markers=['D','^'], ms=16)
fig, ax = plt.subplots(figsize=(20,6))

fig = sm.graphics.interaction_plot(x=df['marital.status'], response=df[">50K"], trace=df.sex,

                                   ax=ax, plottype='both',colors=['coral','lightblue'], markers=['D','^'], ms=16)
fig, ax = plt.subplots(figsize=(26,8))

fig = sm.graphics.interaction_plot(x=df['occupation'], response=df[">50K"], trace=df.sex,

                                   ax=ax, plottype='both',colors=['coral','lightblue'], 

                                   markers=['D','^'], ms=16)
fig, ax = plt.subplots(figsize=(20,6))

fig = sm.graphics.interaction_plot(x=df['marital.status'], response=df[">50K"], trace=df.race,

                                   ax=ax,plottype='both', ms=16)
fig, ax = plt.subplots(figsize=(20,6))

fig = sm.graphics.interaction_plot(x=df['age'], response=df[">50K"], trace=df.sex,

                                   ax=ax, plottype='both',colors=['coral','lightblue'], 

                                   markers=['D','^'], ms=16)
fig, ax = plt.subplots(figsize=(20,6))

fig = sm.graphics.interaction_plot(x=df['hours.per.week'], response=df[">50K"], trace=df.sex,

                                   ax=ax, plottype='both',colors=['coral','lightblue'], markers=['D','^'], ms=16)
fig, ax = plt.subplots(figsize=(20,6))

fig = sm.graphics.interaction_plot(x=df['relationship'], response=df[">50K"], trace=df.sex,

                                   ax=ax, plottype='both', colors=['coral','lightblue'], markers=['D','^'], ms=16)
df['discrete_age']=pd.cut(df.age, [0,30,61], include_lowest=False)
df['discrete_advan_education']=pd.cut(df['education.num'], [0,11,17], include_lowest=False)
df['Husband']=pd.get_dummies(df.relationship).iloc[:,0]
df['Male']=pd.get_dummies(df.sex).iloc[:,1:2]
df['White']=pd.get_dummies(df.race).iloc[:,-1:]
df['age_(30,61]']=pd.get_dummies(df.discrete_age).iloc[:,-1:]
df['advan_education>=12']=pd.get_dummies(df.discrete_advan_education).iloc[:,1:]
df['advan_education>=12']=pd.get_dummies(df.discrete_advan_education).iloc[:,1:]
df['Married-AF-spouse']=pd.get_dummies(df['marital.status']).iloc[:,1:2] 
df['Married-civ-spouse']=pd.get_dummies(df['marital.status']).iloc[:,2:3] 
df['Married-civ or AF-spouse']=df['Married-AF-spouse']+df['Married-civ-spouse']
df['inter_MWAEHM'] = df['Male']*df['White']*df['age_(30,61]']*df['Married-civ or AF-spouse']*df['advan_education>=12']*df['Husband']
fig = px.pie(df, 'inter_MWAEHM', title='MAN WHITE AGE (30,61) MARRIED HUSBAND Ad.EDUCATION').update_traces(hoverinfo='value', textinfo='percent+value', textfont_size=20)

fig.show()
#The groupby method allows us to group the entire population with respect to several variables

df.groupby('inter_MWAEHM').aggregate(np.mean)
fig = px.imshow(df.corr(),x=list(df.corr().columns),y=list(df.corr().columns),width=900,

                height=700,title='Correlation Matrix', color_continuous_scale=px.colors.diverging.Tropic)

fig.show()
df_=df.copy()
df_['native.country'].value_counts()
df_['native_USA']=pd.get_dummies(df_['native.country']).iloc[:,-5:-4]
df_.columns
df_.drop(columns=['education','native.country','discrete_age', 'discrete_advan_education', 'Male', 'White',

       'age_(30,61]', 'advan_education>=12', 'Husband', 'Married-AF-spouse',

       'Married-civ-spouse', 'Married-civ or AF-spouse', 'inter_MWAEHM'],inplace=True)
X=df_['>50K']
df_.drop(columns=['>50K'],inplace=True)
df_=pd.get_dummies(df_)
df_=df_.apply(pd.to_numeric)
df_
# We scale our Matrix (DataSet) First we transform our DF in a scaled Matrix (scaled by the μ and σ of each columm) 

df_=scale(df_)
# from sklearn we import PCA module and we fit our DataSet, we specify the number of PC and call the fit() method 



from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(df_)
# Now we can transform our DataSet

df_2d = pca.transform(df_)
# We generate a new Dataframe with our two components as variables

df_2d = pd.DataFrame(df_2d)

df_2d.columns = ['PC1','PC2']
df_2d['>50K']=X
df_2d
fig = px.scatter(df_2d, x='PC1',y='PC2', title="PC1 vs PC2 scatter plot",

                 color='>50K').update_traces(dict(marker_line_width=1,marker_line_color="black"))

fig.show()
pca = PCA(n_components=3)

pca.fit(df_)
# Now we can transform our DataSet

df_3d = pca.transform(df_)
# We generate a new Dataframe with our two components as variables

df_3d = pd.DataFrame(df_3d)

df_3d.columns = ['PC1','PC2','PC3']
df_3d['>50K']=X
df_3d.corr()
fig = px.scatter_3d(df_3d, x='PC1',y='PC2',z='PC3',title="PC1 vs PC2 vs PC3 scatter plot",

                    color='>50K').update_traces(dict(marker_line_width=1, marker_line_color="black"))

fig.show()
df1=df.groupby('education').aggregate(np.mean)
df1=df1.sort_values(by='>50K')
df1.style.bar(subset=['>50K'],color='#d65f5f')
fig = px.bar(df1, x='education.num', y='>50K',title='Mean of observations (Income >50K 1: Yes) by education',color='hours.per.week', color_continuous_scale=px.colors.diverging.BrBG)

fig.show()
df2=df.iloc[:,:-7].groupby('native.country').aggregate(np.mean)
df2['native.country']=df2.index
fig = px.treemap(df2, path=['native.country', '>50K'], values='>50K')

fig.show()
df2=df2.sort_values(by="education.num", ascending=True,)
fig = px.bar(df2, x='native.country', y='education.num',title='Education num by native-country',color='>50K', color_continuous_scale=px.colors.diverging.BrBG)

fig.show()
df2=df2.sort_values(by="hours.per.week", ascending=True)
fig = px.bar(df2, x='native.country', y='hours.per.week',title='Hour per week by native-country',color='>50K', color_continuous_scale=px.colors.diverging.BrBG)

fig.show()
df3=df.iloc[:,:-9].groupby('sex').aggregate(['mean','median','max','min','std'])
df3.round(2).T
df4=df.iloc[:,:-9].groupby('marital.status').aggregate(np.mean)
df4['marital.status']=df4.index
df4=df4.sort_values(by="age", ascending=True)
df4.style.background_gradient(cmap='Blues')
fig = px.bar(df4, x='marital.status', y='age',title='Capital Gain by Relationship status',

             color='>50K', color_continuous_scale=px.colors.diverging.BrBG)

fig.show()
df5=df.iloc[:,:].groupby('relationship').aggregate(np.mean)
df5=df5.sort_values(by="capital.gain", ascending=True)
df5['relationship']=df5.index
df5.style.background_gradient(cmap='Blues')
fig = px.bar(df5, x='relationship', y='capital.gain',title='Capital Gain by Relationship status',color='>50K', color_continuous_scale=px.colors.diverging.BrBG)

fig.show()
df6=df.iloc[:,:].groupby('occupation').aggregate(np.mean)
df6.round(2)
df6=df6.sort_values(by=">50K", ascending=True)
df6['occupation']=df6.index
fig = px.scatter(df6, x='occupation',y='>50K', title="mean >50K income by occupation",

                 color='age',size='advan_education>=12', color_continuous_scale='tealrose').update_traces(dict(marker_line_width=1,marker_line_color="black"))

fig.show()