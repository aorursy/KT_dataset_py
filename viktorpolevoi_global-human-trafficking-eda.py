import pandas as pd
import numpy as np
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
!pip install --quiet pycountry_convert
from pycountry_convert import country_alpha2_to_country_name, country_name_to_country_alpha3
import os
data = pd.read_csv('../input/global-human-trafficking/human_trafficking.csv')
data[:5]
data.replace('-99', np.nan, inplace=True)
data.replace(-99, np.nan, inplace=True)
def get_alpha3(col):
    try:
        iso_3 =  country_name_to_country_alpha3(col)
    except:
        iso_3 = 'Unknown'
    return iso_3

def get_name(col):
    try:
        name =  country_alpha2_to_country_name(col)
    except:
        name = 'Unknown'
    return name
data['country'] = data['citizenship'].apply(lambda x: get_name(x))
data['alpha_3'] = data['country'].apply(lambda x: get_alpha3(x))
data_map = pd.DataFrame(data.groupby(['country', 'alpha_3'])['alpha_3'].agg(Victims='count')).reset_index()
fig = px.choropleth(data_map, locations="alpha_3",
                    color="Victims",
                    hover_name="country",
                    color_continuous_scale='Viridis_r')
fig.update_layout(title_text="Human Trafficking Victims")
fig.show()
cm = sns.light_palette("red", as_cmap=True)
table = pd.pivot_table(data, values='Datasource', index='country',
                    columns='yearOfRegistration', aggfunc='count', fill_value=0)
table.style.background_gradient(cmap=cm)
data['Victims'] = 1
fig = px.sunburst(data[data.ageBroad.notna()], path=['gender', 'ageBroad'], values='Victims', color='gender',
                  title='Gender and Age of Human Trafficking Victims')
fig.update_layout(width=600, height=600)
fig.show()
data_bar_mg = pd.DataFrame(data.groupby(['gender', 'majorityStatus'])['majorityStatus'].agg(Victims='count')).reset_index()
fig = px.bar(data_bar_mg, x="majorityStatus", y="Victims", color="gender", 
            title="Majority and Gender of Human Trafficking Victims",
            labels={'majorityStatus':'Majority Status'})
fig.update_traces(texttemplate='%{value}', textposition='outside')
fig.update_layout(hovermode='x')
fig.show()
data['meansOfControlConcatenated'] = data['meansOfControlConcatenated'].str.replace('Abuse', 'abuse', regex=True)
data_bar_f = data[(data.meansOfControlConcatenated.notna()) & (data.gender == 'Female')].meansOfControlConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0)
data_bar_m = data[(data.meansOfControlConcatenated.notna()) & (data.gender == 'Male')].meansOfControlConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0)
fig = go.Figure(data=[
    go.Bar(name='Female', x=data_bar_f.index, y=data_bar_f),
    go.Bar(name='Male', x=data_bar_m.index, y=data_bar_m)
])
fig.update_traces(texttemplate='%{value}', textposition='outside')
fig.update_layout(hovermode='x', title_text='Means of Control')
fig.show()
table2 = pd.DataFrame()
for i in data[data.ageBroad.notna()].ageBroad.unique():
    age_col = pd.DataFrame(data[(data.meansOfControlConcatenated.notna()) & (data.ageBroad == i)].meansOfControlConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0))
    age_col.rename(columns={0: i}, inplace=True)
    table2 = pd.concat([table2,age_col],axis=1)

age_list = ['0--8', '9--17', '18--20', '21--23', '24--26', '27--29', '30--38', '39--47', '48+']
table2 = table2.reindex(columns=age_list)
table2.fillna(0).style.background_gradient(cmap=cm).format('{:,.0f}')
data_bar_f = data[(data.typeOfExploitConcatenated.notna()) & (data.gender == 'Female')].typeOfExploitConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0)
data_bar_m = data[(data.typeOfExploitConcatenated.notna()) & (data.gender == 'Male')].typeOfExploitConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0)
fig = go.Figure(data=[
    go.Bar(name='Female', x=data_bar_f.index, y=data_bar_f),
    go.Bar(name='Male', x=data_bar_m.index, y=data_bar_m)
])
fig.update_traces(texttemplate='%{value}', textposition='outside')
fig.update_layout(title_text='Type of Exploit')
fig.show()
table3 = pd.DataFrame()
for i in data[data.ageBroad.notna()].ageBroad.unique():
    age_col = pd.DataFrame(data[(data.typeOfExploitConcatenated.notna()) & (data.ageBroad == i)].typeOfExploitConcatenated.apply(lambda x: pd.value_counts(str(x).split(";"))).sum(axis = 0))
    age_col.rename(columns={0: i}, inplace=True)
    table3 = pd.concat([table3,age_col],axis=1)
    
table3 = table3.reindex(columns=age_list)
table3.fillna(0).style.background_gradient(cmap=cm).format('{:,.0f}')
data_sex_type = data.typeOfSexConcatenated.value_counts()
fig = px.pie(data_sex_type, values=data_sex_type, names=data_sex_type.index,
            title="Distribution of Sex Exploit")
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
data_bar_f = data[(data.RecruiterRelationship.notna()) & (data.gender == 'Female')].RecruiterRelationship.apply(lambda x: pd.value_counts(str(x).split("; "))).sum(axis = 0)
data_bar_m = data[(data.RecruiterRelationship.notna()) & (data.gender == 'Male')].RecruiterRelationship.apply(lambda x: pd.value_counts(str(x).split("; "))).sum(axis = 0)
fig = go.Figure(data=[
    go.Bar(name='Female', x=data_bar_f.index, y=data_bar_f),
    go.Bar(name='Male', x=data_bar_m.index, y=data_bar_m)
])
fig.update_traces(texttemplate='%{value}', textposition='outside')
fig.update_layout(title_text='Recruiter Relationship')
fig.show()
