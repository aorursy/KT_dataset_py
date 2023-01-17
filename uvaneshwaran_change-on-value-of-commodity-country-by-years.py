# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import plotly.graph_objects as go

import plotly.express as px

# Create random data with numpy

import numpy as np

from functools import reduce

from functools import partial 

# Any results you write to the current directory are saved as output.
exp=pd.read_csv('/kaggle/input/india-trade-data/2018-2010_export.csv')
imp=pd.read_csv('/kaggle/input/india-trade-data/2018-2010_import.csv')
imp.groupby(['country','year']).value.sum().reset_index().sort_values(by='value',ascending=False).head(10)
exp.groupby(['country','year']).value.sum().reset_index().sort_values(by='value',ascending=False).head(10)
imp['year'].value_counts()
exp_cou_year=exp['year'].value_counts().reset_index(name=('count')).rename(columns={'index':'year'})

imp_cou_year=imp['year'].value_counts().reset_index(name=('count')).rename(columns={'index':'year'})

exp_cou_year=exp_cou_year.sort_values(by='year')

imp_cou_year=imp_cou_year.sort_values(by='year')
fig = go.Figure()

fig.add_trace(go.Scatter(x=exp_cou_year['year'], y=exp_cou_year['count'],mode='lines+markers',name='export'))

fig.add_trace(go.Scatter(x=imp_cou_year['year'], y=imp_cou_year['count'],mode='lines+markers',name='import'))

fig.update_layout(title_text='Import & Export Counts Based on year') 

fig.show()
def trade(data,column):

    years=data['year'].unique()

    lst=[]

    lst1=[]

    for year in years:

        value1=data[data['year']==year][column].value_counts().reset_index(name='count').rename(columns={'index':column})[:1]

        value1['year']=year

        lst.append(value1)

        value2=data[data['year']==year][column].value_counts().reset_index(name='count').rename(columns={'index':column})[-1:]

        value2['year']=year

        lst1.append(value2)

    value1=pd.concat(lst)

    value1=value1.sort_values(by='year')

    #print(value1)

    fig = px.bar(value1, x='year', y='count',hover_data=[column],color='year', height=300,width=600)

    fig.update_layout(title_text='Top 1 by year')

    fig.show()

    value2=pd.concat(lst1)

    value2=value2.sort_values(by='year')

    #print(value2)

    fig = px.bar(value2, x='year', y='count',hover_data=[column],color='year', height=300,width=600)

    fig.update_layout(title_text='Least 1 by year')

    fig.show()
imp_removed_unspecified=imp[imp['country']!='UNSPECIFIED']

exp_removed_unspecified=exp[exp['country']!='UNSPECIFIED']
trade(imp,'Commodity')
trade(exp,'Commodity')
def year_wise_increase(data,column):

    df=pd.pivot_table(data, values='value', index=[column],columns=['year'], aggfunc=np.sum).reset_index()

    df['d']=df[2011]-df[2010]

    df=df[df['d']>0]

    df['d']=df[2012]-df[2011]

    df=df[df['d']>0]

    df['d']=df[2013]-df[2012]

    df=df[df['d']>0]

    df['d']=df[2014]-df[2013]

    df=df[df['d']>0]

    df['d']=df[2015]-df[2014]

    df=df[df['d']>0]

    df['d']=df[2016]-df[2015]

    df=df[df['d']>0]

    df['d']=df[2017]-df[2016]

    df=df[df['d']>0]

    df['d']=df[2018]-df[2017]

    df=df[df['d']>0]

    df=df.drop(['d'],axis=1)

    return df
exported=year_wise_increase(exp,'country')

exported
exported=exported.drop(['country'],axis=1)

exported=exported.transpose().reset_index()

exported.columns=['Year','Albania','Jamaica','Mexico']
fig = go.Figure()

fig.add_trace(go.Scatter(x=exported['Year'], y=exported['Albania'],mode='lines+markers',name='ALBANIA'))

fig.add_trace(go.Scatter(x=exported['Year'], y=exported['Jamaica'],mode='lines+markers',name='JAMAICA'))

fig.add_trace(go.Scatter(x=exported['Year'], y=exported['Mexico'],mode='lines+markers',name='MEXICO'))

fig.show()
imported=year_wise_increase(imp,'country')

imported
def year_wise_decrease(data,column):

    df=pd.pivot_table(data, values='value', index=[column],columns=['year'], aggfunc=np.sum).reset_index()

    df['d']=df[2011]-df[2010]

    df=df[df['d']<0]

    df['d']=df[2012]-df[2011]

    df=df[df['d']<0]

    df['d']=df[2013]-df[2012]

    df=df[df['d']<0]

    df['d']=df[2014]-df[2013]

    df=df[df['d']<0]

    #df['d']=df[2015]-df[2014]

    #df=df[df['d']<0]

    #df['d']=df[2016]-df[2015]

    #df=df[df['d']<0]

    #df['d']=df[2017]-df[2016]

    #df=df[df['d']<0]

    #df['d']=df[2018]-df[2017]

    #df=df[df['d']<0]

    df=df.drop(['d'],axis=1)

    return df
year_wise_decrease(imp,'country')
year_wise_decrease(exp,'country')
exp_val_year=exp.groupby(['year']).value.sum().reset_index()

imp_val_year=imp.groupby(['year']).value.sum().reset_index()

exp_val_year.columns=['year','export value']

imp_val_year.columns=['year','import value']

exp_imp=exp_val_year.merge(imp_val_year,on='year')

exp_imp['Difference']=exp_imp['export value']-exp_imp['import value']

exp_imp
commodity_exp=year_wise_increase(exp,'Commodity')

commodity_exp
commodity_exp=commodity_exp.drop(['Commodity'],axis=1)

commodity_exp=commodity_exp.transpose().reset_index()

commodity_exp.columns=['Year','CERAMIC PRODUCTS.','FURNITURE; BEDDING, MATTRESSES, MATTRESS SUPPORTS, CUSHIONS AND SIMILAR STUFFED FURNISHING; LAMPS AND LIGHTING FITTINGS NOT ELSEWHERE SPECIFIED OR INC','OPTICAL, PHOTOGRAPHIC CINEMATOGRAPHIC MEASURING, CHECKING PRECISION, MEDICAL OR SURGICAL INST. AND APPARATUS PARTS AND ACCESSORIES THEREOF','PHARMACEUTICAL PRODUCTS']
fig = go.Figure()

fig.add_trace(go.Scatter(x=commodity_exp['Year'], y=commodity_exp['CERAMIC PRODUCTS.'],mode='lines+markers',name='CERAMIC PRODUCTS.'))

fig.add_trace(go.Scatter(x=commodity_exp['Year'], y=commodity_exp['FURNITURE; BEDDING, MATTRESSES, MATTRESS SUPPORTS, CUSHIONS AND SIMILAR STUFFED FURNISHING; LAMPS AND LIGHTING FITTINGS NOT ELSEWHERE SPECIFIED OR INC'],mode='lines+markers',name='FURNITURE; BEDDING, MATTRESSES, MATTRESS SUPPORTS, CUSHIONS AND SIMILAR STUFFED FURNISHING; LAMPS AND LIGHTING FITTINGS NOT ELSEWHERE SPECIFIED OR INC'))

fig.add_trace(go.Scatter(x=commodity_exp['Year'], y=commodity_exp['OPTICAL, PHOTOGRAPHIC CINEMATOGRAPHIC MEASURING, CHECKING PRECISION, MEDICAL OR SURGICAL INST. AND APPARATUS PARTS AND ACCESSORIES THEREOF'],mode='lines+markers',name='OPTICAL, PHOTOGRAPHIC CINEMATOGRAPHIC MEASURING, CHECKING PRECISION, MEDICAL OR SURGICAL INST. AND APPARATUS PARTS AND ACCESSORIES THEREOF'))

fig.add_trace(go.Scatter(x=commodity_exp['Year'], y=commodity_exp['PHARMACEUTICAL PRODUCTS'],mode='lines+markers',name='PHARMACEUTICAL PRODUCTS'))



fig.show()
commodity_imp=year_wise_increase(imp,'Commodity')

commodity_imp
year_wise_decrease(exp,'Commodity')
year_wise_decrease(imp,'Commodity')