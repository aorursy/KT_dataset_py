import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pandas_profiling import ProfileReport
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
data=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
ProfileReport(data)
data.head()
fig = px.scatter(data, x='mba_p', y='salary')
fig.update_layout(title='MBA percentage vs salary',xaxis_title="MBA % ",yaxis_title="Salary")
fig.show()
fig = px.scatter(data, x='mba_p', y='salary',color='specialisation')
fig.update_layout(title='MBA percentage vs salary',xaxis_title="MBA % ",yaxis_title="Salary")
fig.show()
df=data.groupby('specialisation')['mba_p'].mean()
df=pd.DataFrame(df).rename(columns={'mba_p': 'avg. mba %'}).reset_index()
df
fig = go.Figure([go.Pie(labels=df['specialisation'], values=df['avg. mba %'])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,insidetextorientation='radial')

fig.update_layout(title="Avg. %  marks of mba by specialisation",title_x=0.5)
fig.show()

df=pd.DataFrame(data.groupby(['gender','specialisation','status'])['sl_no'].count()).rename(columns={'sl_no': 'no. of students'}).reset_index()
fig = px.sunburst(df, path=['gender','status','specialisation'], values='no. of students')
fig.update_layout(title="Placement % of mba in each specialisation by gender ",title_x=0.5)
fig.show()
mba_percentage=data['mba_p'].values
fig = go.Figure(go.Box(y=mba_percentage,name="MBA %"))
fig.update_layout(title="MBA percentage distribution")
fig.show()
mba_p_1=data[data['specialisation']=="Mkt&Fin"]['mba_p']
mba_p_2=data[data['specialisation']=="Mkt&HR"]['mba_p']    
fig = go.Figure()
fig.add_trace(go.Box(y=mba_p_1,
                     marker_color="blue",
                     name="Mkt&Fn %"))
fig.add_trace(go.Box(y=mba_p_2,
                     marker_color="red",
                     name="Mkt&HR %"))
fig.update_layout(title="Distribution of percentage marks for specialisation -Mkt%Fn &  MKt&HR ")
fig.show()
fig = go.Figure(data=[go.Histogram(x=data['salary'],  # To get Horizontal plot ,change axis - y=campus_computer
                                  marker_color="chocolate",
                      xbins=dict(
                      start=200000, #start range of bin
                      end=1000000,  #end range of bin
                      size=10000    #size of bin
                      ))])
fig.update_layout(title="Distribution of Salary",xaxis_title="Salary",yaxis_title="Counts")
fig.show()
mba_sal=data['salary'].values
fig = go.Figure(go.Box(y=mba_sal,name="salary"))
fig.update_layout(title="Salary distribution")
fig.show()
mba_sal_1=data[data['specialisation']=="Mkt&Fin"]['salary']
mba_sal_2=data[data['specialisation']=="Mkt&HR"]['salary']    
fig = go.Figure()
fig.add_trace(go.Box(y=mba_sal_1,
                     marker_color="blue",
                     name="Salary of Mkt&Fn"))
fig.add_trace(go.Box(y=mba_sal_2,
                     marker_color="red",
                     name="Salary of Mkt&HR"))
fig.update_layout(title="Distribution of salary for specialisation : Mkt&Fn &  MKt&HR ")
fig.show()
fig = go.Figure()
fig.add_trace(go.Histogram(x=mba_sal_1,marker_color="green",name="Mkt&Fn"))
fig.add_trace(go.Histogram(x=mba_sal_2,marker_color="orange",name="MKt&HR"))

# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.update_layout(title="Distribution of salary for specialisation : Mkt&Fn &  MKt&HR",xaxis_title="salary",yaxis_title="Counts")
fig.show()
fig = go.Figure(go.Histogram2d(
        x=data['etest_p'],
        y=data['degree_p']
    ))
fig.update_layout(title='Density of Interview Test & Degree Percentage',xaxis_title="Test Percentage",yaxis_title="Degree Percentage")
fig.show()
degree_percentage=data['degree_p'].values
fig = go.Figure(go.Box(y=degree_percentage,name="MBA %"))
fig.update_layout(title="Degree percentage distribution")
fig.show()
fig = go.Figure(data=[go.Histogram(x=data['degree_p'],marker_color="chocolate")])
fig.update_layout(title="Distribution of degree %",xaxis_title="degree %")
fig.show()
etest_percentage=data['etest_p'].values
fig = go.Figure(go.Box(y=etest_percentage,name="etest %"))
fig.update_layout(title="E-test percentage distribution")
fig.show()
fig = go.Figure(data=[go.Histogram(x=data['etest_p'],marker_color="blue")])
fig.update_layout(title="Distribution of etest %",xaxis_title="etest %")
fig.show()
fig = go.Figure([go.Pie(labels=data['degree_t'].unique(), values=data['degree_t'].value_counts())])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,insidetextorientation='radial')

fig.update_layout(title=" Ratio of students in each field of degree education",title_x=0.5)
fig.show()

sci_tech=data[data['degree_t']=='Sci&Tech']['degree_p']
comm_mgmt=data[data['degree_t']=='Comm&Mgmt']['degree_p']
others=data[data['degree_t']=='Others']['degree_p']

fig = go.Figure()
fig.add_trace(go.Box(y=sci_tech,
                     marker_color="blue",
                     name="science & tech"))
fig.add_trace(go.Box(y=comm_mgmt,
                     marker_color="red",
                     name="commerce and management"))
fig.add_trace(go.Box(y=others,
                     marker_color="green",
                     name="others"))
fig.update_layout(title="percentage distribution in each field of degree education")
fig.show()
hist_data = [sci_tech,comm_mgmt,others] # Added more distplot
group_labels = ['science and tech',"Commerce and management","others"]
colors=['blue',"green","orange"]
fig = ff.create_distplot(hist_data, group_labels,show_hist=False, # Set False to hide histogram bars
                         colors=colors,bin_size=[10000,10000,10000])
fig.update_layout(title="percentage distribution in each field of degree education")
fig.show()
fig = px.scatter(data, x='degree_p', y='salary')
fig.update_layout(title='Degree percentage vs salary',xaxis_title="Degree % ",yaxis_title="Salary")
fig.show()
fig = px.scatter(data, x='mba_p', y='etest_p')
fig.update_layout(title='Degree % vs etest % ',xaxis_title="Degree % ",yaxis_title="Etest %")
fig.show()
df=pd.DataFrame(data.groupby(['workex','degree_t','status'])['sl_no'].count()).rename(columns={'sl_no': 'no. of students'}).reset_index()
fig = px.sunburst(df, path=['workex','status','degree_t'], values='no. of students')
fig.update_layout(title="Placement % of degree in each field  by work experience ",title_x=0.5)
fig.show()
data['salary'] = data['salary'].fillna(0)
sci_tech=data[data['degree_t']=='Sci&Tech']['salary']
comm_mgmt=data[data['degree_t']=='Comm&Mgmt']['salary']
others=data[data['degree_t']=='Others']['salary']
hist_data = [sci_tech,comm_mgmt,others] # Added more distplot
group_labels = ['science and tech',"Commerce and management","others"]
colors=['blue',"green","orange"]
fig = ff.create_distplot(hist_data, group_labels,show_hist=False, # Set False to hide histogram bars
                         colors=colors,bin_size=[10000,10000,10000])
fig.update_layout(title="salary distribution in each field of degree education")
fig.show()
degree=round(data['degree_p'].mean(),2)
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    gauge = {
       'axis': {'range': [None, 100]}},
    value = degree,
    title = {'text': "Average degree %"},
    domain = {'x': [0, 1], 'y': [0, 1]}
))
fig.show()

degree=round(data['mba_p'].mean(),2)
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    gauge = {
       'axis': {'range': [None, 100]}},
    value = degree,
    title = {'text': "Average mba %"},
    domain = {'x': [0, 1], 'y': [0, 1]}
))
fig.show()
fig = px.scatter(data, x='etest_p', y='salary')
fig.update_layout(title=' Etest vs salary ',xaxis_title="Etest % ",yaxis_title="salary")
fig.show()
fig = px.scatter_3d(data, x='ssc_p', y='hsc_p', z='degree_p',
              color='etest_p', size='etest_p', size_max=18,
              symbol='status', opacity=0.7)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
com=data[data['degree_t']=='Comm&Mgmt']
sci=data[data['degree_t']=='Sci&Tech']

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])


fig.add_trace(
    go.Scatter3d(x=com['ssc_p'], y=com['hsc_p'], z=com['degree_p'],name="Commerce"),
    row=1, col=1)


fig.add_trace(
    go.Scatter3d(x=sci['ssc_p'], y=sci['hsc_p'], z=sci['degree_p'],name="Science"),
    row=1, col=2)

fig.update_layout(
    title_text='Percentage scores of Commerce & Science graduates',title_x=0.5)

fig.show()
placed=data[data['status']=='Placed']
fig = go.Figure(data=[go.Mesh3d(x=placed['ssc_p'], y=placed['hsc_p'], z=placed['degree_p'], color='lightblue', opacity=0.50)])
fig.show()
df=pd.DataFrame(data.groupby('gender')['hsc_p','ssc_p'].mean()).reset_index()
fig = go.Figure([go.Pie(labels=df['gender'], values=df['hsc_p'])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,insidetextorientation='radial')

fig.update_layout(title="Avg. %  marks in hsc by gender",title_x=0.5)
fig.show()

fig = go.Figure([go.Pie(labels=df['gender'], values=df['ssc_p'])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=15,insidetextorientation='radial')

fig.update_layout(title="Avg. %  marks of ssc by gender",title_x=0.5)
fig.show()

comm=data[data['hsc_s']=='Commerce']['hsc_p']
sci=data[data['hsc_s']=='Science']['hsc_p']
arts=data[data['hsc_s']=='Arts']['hsc_p']

fig = go.Figure()
fig.add_trace(go.Box(y=comm,
                     marker_color="blue",
                     name="commerce"))
fig.add_trace(go.Box(y=sci,
                     marker_color="red",
                     name="science"))
fig.add_trace(go.Box(y=arts,
                     marker_color="green",
                     name="arts"))
fig.update_layout(title="percentage distribution of different streams in hsc")
fig.show()
others=data[data['hsc_b']=='Others']['hsc_p']
central=data[data['hsc_b']=='Central']['hsc_p']

fig = go.Figure()
fig.add_trace(go.Box(y=others,
                     marker_color="blue",
                     name="others"))
fig.add_trace(go.Box(y=central,
                     marker_color="red",
                     name="central"))

fig.update_layout(title="percentage distribution of different boards in hsc")
fig.show()
others=data[data['ssc_b']=='Others']['ssc_p']
central=data[data['ssc_b']=='Central']['ssc_p']

fig = go.Figure()
fig.add_trace(go.Box(y=others,
                     marker_color="blue",
                     name="others"))
fig.add_trace(go.Box(y=central,
                     marker_color="red",
                     name="central"))

fig.update_layout(title="percentage distribution of different boards in ssc")
fig.show()