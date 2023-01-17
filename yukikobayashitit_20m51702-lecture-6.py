import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)  #In order not to omit displaying array



df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

# print(df)

# print(df.columns)

# print(df['Country/Region'])

# print(df['Country/Region'].values)

# print(np.unique(df['Country/Region'].values))  #Extracting only unique values



#Filtering only South Korea data

selected_country='South Korea'

df = df[df['Country/Region']==selected_country]

df = df.groupby('ObservationDate').sum()  #Grouping by observation date

print(df)
# print(df['Recovered'].diff())



df['daily_confirmed'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recovery'] = df['Recovered'].diff()



# print(df)



# df['daily_confirmed'].plot()

# df['daily_recovery'].plot()

# plt.show()
print(df)
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')



layout_object = go.Layout(title='South Korea daily cases 20M51702',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)

iplot(fig)
fig.write_html('Original_SouthKorea_daily_cases_20M51702.html')
from plotly.subplots import make_subplots





fig2 = go.Figure()

fig2 = make_subplots(specs=[[{"secondary_y": True}]])



daily_recovered_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily recovered')

daily_deaths_object2 = go.Bar(x=df.index,y=df['daily_deaths'].values,name='Daily deaths',opacity=0.5)



fig2.add_trace(

    daily_confirmed_object,

    secondary_y=False,

)



fig2.add_trace(

    daily_recovered_object,

    secondary_y=False,

)



fig2.add_trace(

    daily_deaths_object2,

    secondary_y=True,

)



fig2.update_layout(title_text="South Korea daily cases 20M51702")

fig2.update_xaxes(title_text="Date")

fig2.update_yaxes(title_text="Number of dead people", secondary_y=True)

fig2.update_yaxes(title_text="Number of confirmed/recovered people", secondary_y=False)



fig2.update_layout(

    showlegend=False,

    annotations=[

        dict(

            x="02/18/2020",

            y=0,

            text="Super Spreader",

            showarrow=True,

            arrowhead=7,

            ax=0,

            ay=-300

        ),

        dict(

            x="03/19/2020",

            y=0,

            text="Restrict foreigners' entry",

            showarrow=True,

            arrowhead=7,

            ax=0,

            ay=-300

        ),

        dict(

            x="05/03/2020",

            y=0,

            text="Deregulate the restriction",

            showarrow=True,

            arrowhead=7,

            ax=0,

            ay=-300

        ),

    ]

)





# Please ignore below.

'''

from plotly.subplots import make_subplots



# fig3 = go.Figure()



daily_confirmed_object2 = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed',yaxis='y1')

daily_recovered_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily recovered',yaxis='y1')

daily_deaths_object2 = go.Bar(x=df.index,y=df['daily_deaths'].values,name='Daily deaths',yaxis='y2')



layout = go.Layout(

    title = "South Korea daily cases 20M51702",

    xaxis = dict(title='Date',type='date'),

    yaxis = dict(title="Number of confirmed/recovered people", side='left'),

    yaxis2 = dict(title="Number of dead people",side = 'right',showgrid=False,overlaying='y')

)



fig3 = dict(data = [daily_confirmed_object2, daily_recovered_object, daily_deaths_object2], layout = layout)

iplot(fig3)

'''



iplot(fig2)

fig2.write_html('Modified_SouthKorea_daily_cases_20M51702.html')
df1 = df#[['daily_confirmed']]

df1 = df1.fillna(0.)   #fill na with 0

styled_object = df1.style.background_gradient(cmap='jet').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_20M51702.html','w')  #Save as html

f.write(styled_object.render())
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df1 = df[df['ObservationDate']=='06/12/2020']

df1 = df1.groupby('Country/Region').sum()

df2 = df1.sort_values(by='Confirmed',ascending=False).reset_index() 

df2['Rank'] = df2.index.values+1  #'Rank' indicates global ranking

print(df2)

print(df2[df2['Country/Region']=='South Korea'])

print(df2[df2['Country/Region']=='Japan'])